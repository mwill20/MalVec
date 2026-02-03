"""
MalVec Feature Extractor.

Native implementation of the EMBER feature extraction pipeline.
Extracts 2381 features from a PE file using LIEF.

Based on EMBER 2.0 feature set:
- Byte Histogram (256)
- Byte Entropy Histogram (256)
- String Statistics (104)
- General File Info (10)
- Header Info (62)
- Section Info (255)
- Imports Info (1280)
- Exports Info (128)
- Data Directories (30)
Total: 2381 dimensions

Security:
- Feature extraction runs in sandboxed subprocess
- Timeout enforcement (default 30s)
- Memory limits (default 512MB)
- Crash isolation protects main process
"""

import re
import lief
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Union
from sklearn.feature_extraction import FeatureHasher

from malvec.sandbox import SandboxConfig, SandboxContext, SandboxViolation

# ... (FeatureType classes) ...

class FeatureType:
    """Base class for feature extractors."""
    name = ''
    dim = 0

    def raw_features(self, bytez, lief_binary):
        raise NotImplementedError

    def process_raw_features(self, raw_obj):
        raise NotImplementedError

class ByteHistogram(FeatureType):
    name = 'histogram'
    dim = 256

    def raw_features(self, bytez, lief_binary):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum_val = counts.sum()
        normalized = counts / sum_val if sum_val > 0 else counts
        return normalized

class ByteEntropyHistogram(FeatureType):
    name = 'byteentropy'
    dim = 256
    
    def __init__(self, step=1024, window=2048):
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block):
        # coarse histogram, 16 bytes per bin
        c = np.bincount(block >> 4, minlength=16)
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
        # Shannon entropy
        H = np.sum(-p[wh] * np.log2(p[wh])) * 2 
        Hbin = int(H * 2)
        if Hbin == 16: Hbin = 15
        return Hbin, c

    def raw_features(self, bytez, lief_binary):
        output = np.zeros((16, 16), dtype=int)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            Hbin, c = self._entropy_bin_counts(a)
            output[Hbin, :] += c
        else:
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c
        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum_val = counts.sum()
        normalized = counts / sum_val if sum_val > 0 else counts
        return normalized

class StringExtractor(FeatureType):
    name = 'strings'
    dim = 104

    def __init__(self):
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        self._registry = re.compile(b'HKEY_')
        self._mz = re.compile(b'MZ')

    def raw_features(self, bytez, lief_binary):
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0
            
        return {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'printabledist': c.tolist(),
            'printables': int(csum),
            'entropy': float(H),
            'paths': len(self._paths.findall(bytez)),
            'urls': len(self._urls.findall(bytez)),
            'registry': len(self._registry.findall(bytez)),
            'MZ': len(self._mz.findall(bytez))
        }

    def process_raw_features(self, raw_obj):
        hist_divisor = float(raw_obj['printables']) if raw_obj['printables'] > 0 else 1.0
        return np.hstack([
            raw_obj['numstrings'], raw_obj['avlength'], raw_obj['printables'],
            np.asarray(raw_obj['printabledist']) / hist_divisor, raw_obj['entropy'], raw_obj['paths'], raw_obj['urls'],
            raw_obj['registry'], raw_obj['MZ']
        ]).astype(np.float32)

class GeneralFileInfo(FeatureType):
    name = 'general'
    dim = 10

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return {'size': len(bytez), 'vsize': 0, 'has_debug': 0, 'exports': 0, 'imports': 0, 
                    'has_relocations': 0, 'has_resources': 0, 'has_signature': 0, 'has_tls': 0, 'symbols': 0}
        return {
            'size': len(bytez),
            'vsize': lief_binary.virtual_size,
            'has_debug': int(lief_binary.has_debug),
            'exports': len(lief_binary.exported_functions),
            'imports': len(lief_binary.imported_functions),
            'has_relocations': int(lief_binary.has_relocations),
            'has_resources': int(lief_binary.has_resources),
            'has_signature': int(lief_binary.has_signatures),
            'has_tls': int(lief_binary.has_tls),
            'symbols': len(lief_binary.symbols),
        }

    def process_raw_features(self, raw_obj):
        return np.asarray([
            raw_obj['size'], raw_obj['vsize'], raw_obj['has_debug'], raw_obj['exports'], raw_obj['imports'],
            raw_obj['has_relocations'], raw_obj['has_resources'], raw_obj['has_signature'], raw_obj['has_tls'],
            raw_obj['symbols']
        ], dtype=np.float32)

class HeaderFileInfo(FeatureType):
    name = 'header'
    dim = 62

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj['coff'] = {'timestamp': 0, 'machine': "", 'characteristics': []}
        raw_obj['optional'] = {
            'subsystem': "", 'dll_characteristics': [], 'magic': "",
            'major_image_version': 0, 'minor_image_version': 0,
            'major_linker_version': 0, 'minor_linker_version': 0,
            'major_operating_system_version': 0, 'minor_operating_system_version': 0,
            'major_subsystem_version': 0, 'minor_subsystem_version': 0,
            'sizeof_code': 0, 'sizeof_headers': 0, 'sizeof_heap_commit': 0
        }
        if lief_binary is None: return raw_obj

        raw_obj['coff']['timestamp'] = lief_binary.header.time_date_stamps
        raw_obj['coff']['machine'] = str(lief_binary.header.machine).split('.')[-1]
        raw_obj['coff']['characteristics'] = [str(c).split('.')[-1] for c in lief_binary.header.characteristics_list]
        
        raw_obj['optional']['subsystem'] = str(lief_binary.optional_header.subsystem).split('.')[-1]
        raw_obj['optional']['dll_characteristics'] = [str(c).split('.')[-1] for c in lief_binary.optional_header.dll_characteristics_lists]
        raw_obj['optional']['magic'] = str(lief_binary.optional_header.magic).split('.')[-1]
        raw_obj['optional']['major_image_version'] = lief_binary.optional_header.major_image_version
        raw_obj['optional']['minor_image_version'] = lief_binary.optional_header.minor_image_version
        raw_obj['optional']['major_linker_version'] = lief_binary.optional_header.major_linker_version
        raw_obj['optional']['minor_linker_version'] = lief_binary.optional_header.minor_linker_version
        raw_obj['optional']['major_operating_system_version'] = lief_binary.optional_header.major_operating_system_version
        raw_obj['optional']['minor_operating_system_version'] = lief_binary.optional_header.minor_operating_system_version
        raw_obj['optional']['major_subsystem_version'] = lief_binary.optional_header.major_subsystem_version
        raw_obj['optional']['minor_subsystem_version'] = lief_binary.optional_header.minor_subsystem_version
        raw_obj['optional']['sizeof_code'] = lief_binary.optional_header.sizeof_code
        raw_obj['optional']['sizeof_headers'] = lief_binary.optional_header.sizeof_headers
        raw_obj['optional']['sizeof_heap_commit'] = lief_binary.optional_header.sizeof_heap_commit
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['coff']['timestamp'],
            FeatureHasher(10, input_type="string").transform([[raw_obj['coff']['machine']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['coff']['characteristics']]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['subsystem']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['optional']['dll_characteristics']]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['magic']]]).toarray()[0],
            raw_obj['optional']['major_image_version'], raw_obj['optional']['minor_image_version'],
            raw_obj['optional']['major_linker_version'], raw_obj['optional']['minor_linker_version'],
            raw_obj['optional']['major_operating_system_version'], raw_obj['optional']['minor_operating_system_version'],
            raw_obj['optional']['major_subsystem_version'], raw_obj['optional']['minor_subsystem_version'],
            raw_obj['optional']['sizeof_code'], raw_obj['optional']['sizeof_headers'], raw_obj['optional']['sizeof_heap_commit'],
        ]).astype(np.float32)

class SectionInfo(FeatureType):
    name = 'section'
    dim = 255

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None: return {"entry": "", "sections": []}
        try:
            entry_section = lief_binary.section_from_offset(lief_binary.entrypoint).name
        except:
            entry_section = ""
            for s in lief_binary.sections:
                # Robust check for MEM_EXECUTE execution permission
                # s.characteristics_lists returns list of enums, convert to str to be safe
                chars = [str(c) for c in s.characteristics_lists]
                if any("MEM_EXECUTE" in c for c in chars):
                    entry_section = s.name
                    break
        
        return {"entry": entry_section, "sections": [{
            'name': s.name, 'size': s.size, 'entropy': s.entropy, 'vsize': s.virtual_size,
            'props': [str(c).split('.')[-1] for c in s.characteristics_lists]
        } for s in lief_binary.sections]}

    def process_raw_features(self, raw_obj):
        sections = raw_obj['sections']
        general = [
            len(sections),
            sum(1 for s in sections if s['size'] == 0),
            sum(1 for s in sections if s['name'] == ""),
            sum(1 for s in sections if 'MEM_READ' in s['props'] and 'MEM_EXECUTE' in s['props']),
            sum(1 for s in sections if 'MEM_WRITE' in s['props'])
        ]
        
        # Features hashed
        props = ['name', 'size', 'entropy', 'vsize']
        hashed_features = []
        for p in props:
            # Note: FeatureHasher expects (name, value) pairs or strings
            if p == 'name': continue # handled logic differs from EMBER slightly? No, EMBER hashes name+val
            vals = [(s['name'], s[p]) for s in sections]
            hashed_features.append(FeatureHasher(50, input_type="pair").transform([vals]).toarray()[0])
            
        # Replicating EMBER exactly:
        section_sizes = [(s['name'], s['size']) for s in sections]
        section_entropy = [(s['name'], s['entropy']) for s in sections]
        section_vsize = [(s['name'], s['vsize']) for s in sections]
        
        hashed = [
            FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0],
            FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0],
            FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0],
            FeatureHasher(50, input_type="string").transform([[raw_obj['entry']]]).toarray()[0],
            FeatureHasher(50, input_type="string").transform([[p for s in sections for p in s['props'] if s['name'] == raw_obj['entry']]]).toarray()[0]
        ]
        
        return np.hstack(general + hashed).astype(np.float32)

class ImportsInfo(FeatureType):
    name = 'imports'
    dim = 1280

    def raw_features(self, bytez, lief_binary):
        imports = {}
        if lief_binary is None: return imports
        for lib in lief_binary.imports:
            if lib.name not in imports: imports[lib.name] = []
            for entry in lib.entries:
                if entry.is_ordinal: imports[lib.name].append("ordinal" + str(entry.ordinal))
                else: imports[lib.name].append(entry.name[:10000])
        return imports

    def process_raw_features(self, raw_obj):
        libraries = list(set([l.lower() for l in raw_obj.keys()]))
        libraries_hashed = FeatureHasher(256, input_type="string").transform([libraries]).toarray()[0]
        imports = [lib.lower() + ':' + e for lib, elist in raw_obj.items() for e in elist]
        imports_hashed = FeatureHasher(1024, input_type="string").transform([imports]).toarray()[0]
        return np.hstack([libraries_hashed, imports_hashed]).astype(np.float32)

class ExportsInfo(FeatureType):
    name = 'exports'
    dim = 128

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None: return []
        return [export.name[:10000] for export in lief_binary.exported_functions]

    def process_raw_features(self, raw_obj):
        return FeatureHasher(128, input_type="string").transform([raw_obj]).toarray()[0].astype(np.float32)

class DataDirectories(FeatureType):
    name = 'datadirectories'
    dim = 30 # 15 * 2

    def raw_features(self, bytez, lief_binary):
        output = []
        if lief_binary is None: return output
        for data_directory in lief_binary.data_directories:
            output.append({
                "name": str(data_directory.type).replace("DATA_DIRECTORY.", ""),
                "size": data_directory.size,
                "virtual_address": data_directory.rva
            })
        return output

    def process_raw_features(self, raw_obj):
        features = np.zeros(30, dtype=np.float32)
        # Assuming order is preserved from LIEF standard list, but robust impl checks names
        # Simplified for brevity/performance
        for i, d in enumerate(raw_obj):
            if i < 15:
                features[2*i] = d['size']
                features[2*i+1] = d['virtual_address']
        return features

class FeatureExtractor:
    """
    Extracts EMBERv2 features from PE files.

    Security Features:
    - Optional sandboxed extraction (recommended for untrusted files)
    - Timeout enforcement (default 30s)
    - Memory limits (default 512MB)
    - Crash isolation protects main process

    Usage:
        # With sandboxing (recommended for untrusted files)
        extractor = FeatureExtractor(sandbox=True)
        features = extractor.extract(file_path)

        # Without sandboxing (for trusted files only)
        extractor = FeatureExtractor(sandbox=False)
        features = extractor.extract(file_path)
    """

    def __init__(
        self,
        sandbox: bool = True,
        config: SandboxConfig = None
    ):
        """
        Initialize feature extractor.

        Args:
            sandbox: Enable sandboxed extraction (recommended).
            config: Sandbox configuration. Uses defaults if not provided.
        """
        self.features = [
            ByteHistogram(), ByteEntropyHistogram(), StringExtractor(),
            GeneralFileInfo(), HeaderFileInfo(), SectionInfo(),
            ImportsInfo(), ExportsInfo(), DataDirectories()
        ]
        self.dim = sum(f.dim for f in self.features)  # Should be 2381
        self.sandbox_enabled = sandbox
        self.sandbox_config = config or SandboxConfig()

    def extract(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Extract features from a PE file.

        If sandboxing is enabled, runs extraction in isolated subprocess
        with timeout and memory limits.

        Args:
            file_path: Path to PE file.

        Returns:
            np.ndarray: Feature vector of shape (2381,).

        Raises:
            SandboxViolation: If sandbox constraints are violated.
            RuntimeError: If extraction fails.
            FileNotFoundError: If file doesn't exist.
        """
        file_path = Path(file_path)

        if self.sandbox_enabled:
            return self._extract_sandboxed(file_path)
        else:
            return self._extract_impl(file_path)

    def _extract_sandboxed(self, file_path: Path) -> np.ndarray:
        """
        Run extraction in sandboxed subprocess.

        Args:
            file_path: Path to PE file.

        Returns:
            Feature vector.

        Raises:
            SandboxViolation: On timeout, memory limit, or crash.
        """
        with SandboxContext(self.sandbox_config) as sandbox:
            # Validate file before processing
            sandbox.validate_file(file_path)

            try:
                # Run extraction in isolated process
                features = sandbox.run(extract_in_process, file_path)
                return features
            except SandboxViolation:
                raise
            except Exception as e:
                raise RuntimeError(f"Feature extraction failed: {e}") from e

    def _extract_impl(self, file_path: Path) -> np.ndarray:
        """
        Actual extraction implementation (may run in sandbox).

        Args:
            file_path: Path to PE file.

        Returns:
            Feature vector of shape (2381,).
        """
        with open(file_path, 'rb') as f:
            bytez = f.read()

        try:
            # Use file_path directly for speed and memory efficiency
            lief_binary = lief.PE.parse(str(file_path))
        except (lief.bad_format, lief.bad_file, lief.pe_error, lief.parser_error, RuntimeError) as e:
            # Log but continue with fallback (bytez-only features)
            lief_binary = None
        except Exception:
            raise

        feature_vectors = []
        for feature in self.features:
            raw = feature.raw_features(bytez, lief_binary)
            vec = feature.process_raw_features(raw)
            feature_vectors.append(vec)

        return np.hstack(feature_vectors).astype(np.float32)

def extract_in_process(file_path):
    """
    Helper function to be used with isolation.run_isolated.
    Instantiates the FeatureExtractor and extracts features.

    This function is designed to run in an isolated subprocess.
    It creates a non-sandboxed extractor since the sandbox
    context handles the isolation.

    Args:
        file_path: Path to the PE file.

    Returns:
        np.ndarray: Extracted features.
    """
    # Create extractor WITHOUT sandboxing since we're already in sandbox
    extractor = FeatureExtractor(sandbox=False)
    return extractor._extract_impl(file_path)
