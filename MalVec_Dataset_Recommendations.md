# MalVec Dataset Recommendations

## 🎯 Top 3 Free Datasets for MalVec

Based on research, here are the best free/low-cost datasets for building and testing MalVec:

---

## 1. EMBER Dataset (RECOMMENDED - PRIMARY)

**Source:** Elastic Security (formerly Endgame)  
**Link:** https://github.com/elastic/ember  
**Cost:** FREE (Apache 2.0 License)

### Why This is Perfect for MalVec:

✅ **Large Scale:** 1.1M PE files (EMBER2017) + 3.2M files (EMBER2024)  
✅ **Labeled:** 300K malicious, 300K benign, 300K unlabeled (training) + 200K test  
✅ **Features Included:** Pre-extracted PE features (imports, sections, entropy, strings)  
✅ **Temporal Split:** Training from early periods, testing from final 2 months (concept drift)  
✅ **Industry Standard:** Most cited malware ML benchmark  
✅ **Active Maintenance:** EMBER2024 just released (Sep 2023-Dec 2024 samples)

### What You Get:

```
ember2018/
├── train_features_0.jsonl  # Training samples (300K malicious)
├── train_features_1.jsonl  # Training samples (300K benign)
├── train_features_2.jsonl  # Training samples (300K unlabeled)
├── test_features.jsonl     # Test samples (200K labeled)
└── ember_model_2018.txt    # Baseline LightGBM model
```

**Features Per Sample:**
- General file info (size, vsize, has_debug, etc.)
- Header info (COFF, optional header)
- Imported functions
- Exported functions
- Section characteristics
- Byte histogram (256 values)
- Byte-entropy histogram
- String extractions

### How to Download:

```bash
# Install ember python package
pip install ember-ml

# Download dataset (will prompt for download location)
python -c "import ember; ember.create_vectorized_features('/data/ember2018/')"
```

**Alternative:** Direct download from AWS S3:
- EMBER 2017 v2: https://ember.elastic.co/ember_dataset_2017_2.tar.bz2
- EMBER 2018: https://ember.elastic.co/ember_dataset_2018_2.tar.bz2

### MalVec Integration Strategy:

**Phase 1 (Baseline):**
- Use pre-extracted features as input to embedder
- Train on 300K malicious + 300K benign
- Test on 200K holdout set
- Compare against EMBER's baseline LightGBM model

**Phase 2 (From Binaries):**
- EMBER only provides hashes, not binaries (to respect copyright)
- Use hashes to download binaries from VirusTotal (requires premium API)
- OR use EMBER features as proxy for actual binary analysis

**Recommended:** Start with EMBER features, then augment with actual binaries later

---

## 2. BODMAS Dataset (SECONDARY - TEMPORAL ANALYSIS)

**Source:** Blue Hexagon (now defunct, dataset lives on)  
**Link:** https://whyisyoung.github.io/BODMAS/  
**Cost:** FREE (requires email request with justification)

### Why This Complements EMBER:

✅ **Temporal Coverage:** 5+ years of samples (enables drift studies)  
✅ **Feature Vectors:** 134,435 samples with 2,381 features each  
✅ **Malware Families:** Family labels included (not just malicious/benign)  
✅ **Metadata Rich:** SHA-256, first-seen timestamp, family name  
✅ **Real Binaries Available:** Authors will share actual malware samples (with justification)

### What You Get:

```
bodmas/
├── bodmas.npz              # Feature vectors (134,435 x 2,381)
└── bodmas_metadata.csv     # SHA-256, timestamp, family
```

**Features:**
- Static PE features (2,381 dimensions)
- Unnormalized (good for tree models, needs normalization for neural nets)
- Chronologically ordered

### How to Download:

1. **Request Access:**
   - Email: liminy2@illinois.edu, zhic4@illinois.edu
   - CC: gangw@illinois.edu
   - Subject: "BODMAS Dataset Request for MalVec Research"
   
2. **In Your Email:**
```
Dear BODMAS Team,

I am requesting access to the BODMAS dataset for use in my MalVec 
research project - a malware detection system using embedding-space 
analysis.

Purpose: Academic/research project to evaluate embedding-based 
detection of polymorphic malware variants.

Google Drive Email: your.email@gmail.com

I agree to:
- Use dataset for research purposes only
- Not redistribute the dataset
- Cite the BODMAS paper in any publications

Thank you,
[Your Name]
```

3. **Load the Data:**
```python
import numpy as np
data = np.load('bodmas.npz')
X = data['X']  # (134435, 2381)
y = data['y']  # 0=benign, 1=malicious
```

### MalVec Integration Strategy:

**Use Cases:**
- Temporal drift analysis (compare embedding stability over time)
- Family classification (cluster embeddings by family)
- Cross-dataset validation (train on EMBER, test on BODMAS)
- Longitudinal studies (track how clusters evolve)

**Recommended:** Use as validation set after primary training on EMBER

---

## 3. VirusTotal API + Public Malware Repos (TERTIARY - AUGMENTATION)

**Source:** VirusTotal Public API + Community Repos  
**Link:** https://www.virustotal.com/gui/  
**Cost:** FREE (with rate limits) or $500/month (Premium)

### Why This is Useful:

✅ **Label Enrichment:** Get VirusTotal scores for any hash  
✅ **Real-World Samples:** Download actual binaries (premium API)  
✅ **Fresh Data:** Most recent malware samples  
✅ **Metadata Rich:** AV detections, behavior reports, relationships  
✅ **Community Datasets:** Several free malware repos on HuggingFace

### Free Tier Limitations:

**VirusTotal Public API:**
- 4 requests/minute
- 500 requests/day
- Can check hashes (no download)
- Can upload files for scanning
- Cannot download binaries

**Premium API ($500/month):**
- 1000 requests/minute
- Unlimited daily requests
- Can download binaries
- Live hunt notifications
- Retroactive hunting

### Free Alternatives to VirusTotal Premium:

#### 3a. HuggingFace Malware Datasets

**unileon-robotics/malware-samples:**
- URL: https://huggingface.co/datasets/unileon-robotics/malware-samples
- Real malware binaries (.exe files)
- CAPEv2 sandbox reports (JSON, HTML)
- Screenshots of execution
- Dropped files
- **WARNING:** Contains actual malware - use with caution

**IQSeC-Lab/LAMDA:**
- URL: https://huggingface.co/datasets/IQSeC-Lab/LAMDA
- Android APK malware (2013-2025)
- Features extracted from AndroZoo
- VirusTotal counts included
- Family labels via AVClass2

#### 3b. GitHub Malware Datasets

**theZoo (Malware DB):**
- URL: https://github.com/ytisf/theZoo
- Live malware samples
- Organized by family
- **WARNING:** Actual malware, use in isolated environment

**MalwareBazaar (abuse.ch):**
- URL: https://bazaar.abuse.ch/
- Free malware sample downloads
- Recent samples (last 24h, 7d, 30d)
- Hash-based search
- API available

### How to Use VirusTotal (Free):

```python
# Check hash reputation (free tier)
import requests

VT_API_KEY = "your_free_api_key"  # Get from virustotal.com
hash_to_check = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

url = f"https://www.virustotal.com/api/v3/files/{hash_to_check}"
headers = {"x-apikey": VT_API_KEY}

response = requests.get(url, headers=headers)
data = response.json()

# Extract detection count
detections = data['data']['attributes']['last_analysis_stats']
print(f"Malicious: {detections['malicious']}/{detections['malicious'] + detections['undetected']}")
```

### MalVec Integration Strategy:

**Use Cases:**
- Enrich EMBER hashes with VirusTotal scores (free tier sufficient)
- Validate cluster assignments (do similar embeddings get similar VT scores?)
- Gather fresh samples from MalwareBazaar for testing
- Use HuggingFace datasets for additional training data

**Recommended:** Use free tier for validation, skip premium unless critical

---

## 📊 Recommended Dataset Strategy for MalVec

### Phase 1: MVP (Proof of Concept)

**Primary Dataset:** EMBER 2018  
**Size:** 1.1M samples  
**Timeline:** Week 1-2  

**Goal:** Prove embedding approach works
- Train on EMBER features (not binaries)
- Generate embeddings from feature vectors
- Build FAISS index
- Classify with K-NN
- Compare against EMBER baseline

**Success Metric:** Beat EMBER LightGBM baseline by >5%

---

### Phase 2: Binary Analysis

**Primary Dataset:** EMBER hashes + VirusTotal free tier  
**Size:** ~10K samples (start small)  
**Timeline:** Week 3-4  

**Goal:** Prove feature extraction pipeline works
- Download 10K samples from MalwareBazaar (free)
- Extract features using pefile/LIEF
- Generate embeddings from extracted features
- Compare against EMBER pre-extracted features

**Success Metric:** Feature extraction completes without crashes

---

### Phase 3: Validation

**Primary Dataset:** BODMAS (request access)  
**Size:** 134K samples  
**Timeline:** Week 5-6  

**Goal:** Cross-dataset validation
- Train on EMBER
- Test on BODMAS
- Analyze embedding stability over time
- Family-level clustering analysis

**Success Metric:** >85% accuracy on BODMAS test set

---

### Phase 4: Production

**Primary Dataset:** EMBER + MalwareBazaar + BODMAS  
**Size:** 1M+ samples  
**Timeline:** Week 7+  

**Goal:** Production-ready system
- Combine all datasets
- Continuous retraining
- Fresh sample collection
- Drift monitoring

**Success Metric:** Operational deployment

---

## 🔧 Dataset Download Checklist

### Immediate (Week 1):

- [ ] Download EMBER 2018 dataset
  ```bash
  pip install ember-ml
  python -c "import ember; ember.create_vectorized_features('/data/ember2018/')"
  ```

- [ ] Sign up for VirusTotal free API
  - URL: https://www.virustotal.com/gui/my-apikey
  - Store API key in `.env` file (never commit!)

### Soon (Week 2):

- [ ] Request BODMAS access
  - Send email to BODMAS team
  - Wait for Google Drive access

- [ ] Explore HuggingFace datasets
  - Browse: https://huggingface.co/datasets?search=malware
  - Download samples for testing

### Later (Week 3+):

- [ ] Set up MalwareBazaar scraper (if needed)
- [ ] Consider VirusTotal Premium (if budget allows)

---

## 💰 Cost Breakdown

| Dataset | Cost | Samples | Labels | Binaries |
|---------|------|---------|--------|----------|
| EMBER 2018 | FREE | 1.1M | Yes | No (hashes only) |
| BODMAS | FREE | 134K | Yes | Yes (on request) |
| VirusTotal Free | FREE | Unlimited | Yes | No |
| VirusTotal Premium | $500/mo | Unlimited | Yes | Yes |
| MalwareBazaar | FREE | ~1M | Partial | Yes |
| HuggingFace | FREE | Varies | Varies | Yes |

**Recommended Budget:** $0 to start (EMBER + BODMAS + free sources)

---

## 🛡️ Security Warnings

**When working with malware datasets:**

1. **Never execute samples** - Static analysis only
2. **Isolated environment** - VM or air-gapped system
3. **Encrypt at rest** - Use encrypted volumes
4. **Network isolation** - No internet access from analysis VM
5. **Legal compliance** - Some jurisdictions prohibit malware possession

**For MalVec specifically:**
- All security boundaries in NORTHSTAR.md apply
- Samples processed in sandbox (30s timeout, no network)
- Feature extraction only (no execution)

---

## 📚 Citation Requirements

**If you publish using these datasets:**

**EMBER:**
```bibtex
@article{anderson2018ember,
  title={EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models},
  author={Anderson, Hyrum S and Roth, Phil},
  journal={arXiv preprint arXiv:1804.04637},
  year={2018}
}
```

**BODMAS:**
```bibtex
@inproceedings{yang2021bodmas,
  title={BODMAS: An Open Dataset for Learning based Temporal Analysis of PE Malware},
  author={Yang, Limin and Ciptadi, Arridhana and Laziuk, Ihar and Ahmadzadeh, Ali and Wang, Gang},
  booktitle={Deep Learning and Security Workshop},
  year={2021}
}
```

---

## 🚀 Next Steps

1. **Start with EMBER** - Download and explore
2. **Request BODMAS** - Send access email
3. **Sign up for VirusTotal** - Get free API key
4. **Update BUILD_INSTRUCTIONS.md** - Add dataset download steps
5. **Begin Phase 1** - Proof of concept with EMBER

**You now have everything needed to build MalVec with real data! 🔥**
