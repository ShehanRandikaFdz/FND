# Real-Time Fake News Detector - Quick Reference

## 🎯 TL;DR - 5 Minute Summary

Transform your FND system into a **real-time news monitoring platform** that automatically fetches, analyzes, and alerts you about potentially fake news from international sources.

---

## 📊 What You Get

```
Before: Manual text input → Analyze → Result
After:  Auto-fetch news → Batch analyze → Dashboard + Alerts
```

### Key Features
✅ **Automatic News Fetching** - Every 30 minutes  
✅ **Multi-Source Coverage** - 80,000+ sources worldwide  
✅ **Real-Time Analysis** - Existing ML models (no retraining)  
✅ **Smart Alerts** - Email/Telegram for suspicious content  
✅ **Live Dashboard** - Monitor credibility scores in real-time  

---

## 🚀 Quick Start (30 Minutes)

### Step 1: Get API Keys (10 min)

**NewsAPI.org** (Primary) - FREE
1. Visit: https://newsapi.org/register
2. Enter email → Get API key
3. Free tier: 100 requests/day, 1000 articles/day

**The Guardian** (Secondary) - FREE
1. Visit: https://bonobo.capi.gutools.co.uk/register/developer
2. Register → Get API key
3. Free tier: 5,000 requests/day

### Step 2: Install Dependencies (5 min)

```powershell
pip install newsapi-python requests feedparser python-dotenv SQLAlchemy APScheduler
```

### Step 3: Configure (5 min)

Create `.env` file:
```env
NEWSAPI_KEY=your_key_here
GUARDIAN_API_KEY=your_key_here
FETCH_INTERVAL_MINUTES=30
```

### Step 4: Test Connection (5 min)

```python
# test_api.py
from newsapi import NewsApiClient

client = NewsApiClient(api_key='YOUR_KEY')
headlines = client.get_top_headlines(country='us', page_size=5)

for article in headlines['articles']:
    print(f"✅ {article['title']}")
```

### Step 5: Run (5 min)

```powershell
python realtime/scheduler.py
```

**Done!** Your system now auto-monitors news 24/7 🎉

---

## 📋 Architecture Overview

```
┌─────────────────────────────────────────┐
│   NEWS SOURCES (APIs)                   │
│   • NewsAPI (80k sources)               │
│   • The Guardian (quality journalism)   │
│   • RSS Feeds (100% free backup)       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   NEWS FETCHER (Every 30 min)          │
│   • Fetch latest articles               │
│   • Deduplicate by URL/content         │
│   • Save to database                    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   ANALYZER (Every 5 min)               │
│   • Load pending articles               │
│   • Run through SVM/LSTM/BERT          │
│   • Calculate credibility score         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   ALERTS & DASHBOARD                    │
│   • Score < 0.4 → HIGH alert           │
│   • Score 0.4-0.6 → MEDIUM alert       │
│   • Live dashboard with filters         │
└─────────────────────────────────────────┘
```

---

## 💻 File Structure

```
FND/
├── news_apis/                  # 🆕 API clients
│   ├── newsapi_client.py      # NewsAPI integration
│   ├── guardian_client.py     # Guardian integration
│   └── rss_parser.py          # RSS feed parser
│
├── realtime/                   # 🆕 Real-time processing
│   ├── news_fetcher.py        # Fetch news
│   ├── processor.py           # Analyze articles
│   ├── scheduler.py           # Run 24/7
│   └── alert_system.py        # Send alerts
│
├── database/                   # 🆕 Data storage
│   ├── models.py              # Database schema
│   └── db_manager.py          # CRUD operations
│
├── dashboard/                  # 🆕 Real-time UI
│   └── realtime_app.py        # Live monitoring dashboard
│
└── [existing files unchanged]
```

---

## 🌐 Best News APIs

| API | Free Tier | Coverage | Recommendation |
|-----|-----------|----------|----------------|
| **NewsAPI.org** | 1000 articles/day | 80,000 sources | ⭐ START HERE |
| **The Guardian** | 5000 requests/day | Guardian archive | ⭐ ADD THIS |
| **GNews** | 100 requests/day | 60,000 sources | Good alternative |
| **RSS Feeds** | Unlimited | Major outlets | Free backup |
| **Reuters/AP** | Enterprise only | Premium quality | Scale later |

### Recommended Combo (FREE)
1. **NewsAPI** - Broad international coverage
2. **The Guardian** - Quality journalism
3. **RSS Feeds** - Backup + niche sources

**Total Cost:** $0/month  
**Coverage:** 80,000+ sources  
**Requests:** 6,000+ per day

---

## 📊 Implementation Timeline

### **Week 1-2: Foundation** ⏱️ 10-15 hours
- [ ] Register for APIs (30 min)
- [ ] Create `news_apis/` module (3 hours)
- [ ] Build basic fetcher (4 hours)
- [ ] Setup database (3 hours)
- [ ] Test end-to-end (2 hours)

**Deliverable:** Basic news fetching working

### **Week 3-4: Real-Time Processing** ⏱️ 15-20 hours
- [ ] Build processor pipeline (5 hours)
- [ ] Implement scheduling (3 hours)
- [ ] Add deduplication (4 hours)
- [ ] Create alert system (4 hours)
- [ ] Testing & optimization (4 hours)

**Deliverable:** 24/7 automated monitoring

### **Week 5-6: Dashboard & Polish** ⏱️ 10-15 hours
- [ ] Build live dashboard (6 hours)
- [ ] Add filters & search (3 hours)
- [ ] Performance optimization (3 hours)
- [ ] Documentation (2 hours)

**Deliverable:** Production-ready system

**Total Time:** 35-50 hours (6-8 weeks part-time)

---

## 💰 Cost Breakdown

### Free Tier (Months 1-3)
```
NewsAPI Free:        $0
Guardian API:        $0
Database (SQLite):   $0
Hosting (local):     $0
─────────────────────────
TOTAL:              $0/month
```

### Starter Tier (Months 3-6)
```
NewsAPI Business:    $449/month
Guardian API:        $0
PostgreSQL:          $15/month
Cloud hosting:       $20/month
─────────────────────────
TOTAL:              $484/month
```

### Production Tier (6+ months)
```
Premium APIs:        $500-1000/month
Database + Redis:    $50/month
Cloud hosting:       $100/month
Monitoring:          $50/month
─────────────────────────
TOTAL:              $700-1200/month
```

**Recommended Path:** Start free → Prove value → Scale gradually

---

## 🔧 Code Templates

### 1. Basic API Test (5 lines)

```python
from newsapi import NewsApiClient
client = NewsApiClient(api_key='YOUR_KEY')
headlines = client.get_top_headlines(country='us', page_size=10)
for article in headlines['articles']:
    print(article['title'])
```

### 2. Fetch & Analyze (10 lines)

```python
from news_apis.newsapi_client import NewsAPIClient
from credibility_analyzer.credibility_analyzer import CredibilityAnalyzer

# Fetch
api = NewsAPIClient()
articles = api.get_top_headlines(page_size=10)

# Analyze
analyzer = CredibilityAnalyzer()
for article in articles:
    result = analyzer.analyze_credibility(article['content'])
    print(f"{article['title']}: {result['credibility_score']:.2f}")
```

### 3. Auto-Monitor (Scheduler)

```python
from apscheduler.schedulers.background import BackgroundScheduler
from realtime.news_fetcher import NewsFetcher
from realtime.processor import RealtimeProcessor

scheduler = BackgroundScheduler()
fetcher = NewsFetcher()
processor = RealtimeProcessor()

# Fetch every 30 minutes
scheduler.add_job(fetcher.fetch_all_sources, 'interval', minutes=30)

# Process every 5 minutes
scheduler.add_job(processor.process_pending_articles, 'interval', minutes=5)

scheduler.start()
print("🚀 Monitoring started!")
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: Rate Limit Exceeded
```
Error: You have exceeded your rate limit
```
**Solution:**
- Reduce fetch frequency
- Use multiple API keys (rotate)
- Add Guardian API as backup
- Implement caching

### Issue 2: Empty Results
```
Fetched 0 articles
```
**Solution:**
- Check API key validity
- Verify internet connection
- Try different country/language
- Check API status page

### Issue 3: Processing Too Slow
```
Processing taking 10+ seconds per article
```
**Solution:**
- Enable batch processing
- Use GPU for BERT model
- Implement parallel processing
- Consider lighter models

### Issue 4: Database Errors
```
OperationalError: database is locked
```
**Solution:**
- Switch to PostgreSQL (production)
- Implement connection pooling
- Use write-ahead logging (WAL)
- Add retry logic

---

## 📈 Success Metrics

### Technical KPIs
- ✅ Uptime: >99%
- ✅ Fetch latency: <30 seconds
- ✅ Processing latency: <5 minutes
- ✅ False positive rate: <2%

### Business KPIs
- ✅ Articles processed: 5,000+ daily
- ✅ Sources covered: 50+ international
- ✅ Detection speed: <30 minutes from publication
- ✅ Alert precision: >95% accuracy

---

## 🎯 Decision Tree

```
Do you need real-time monitoring?
├─ YES
│  └─ Daily article volume?
│     ├─ < 1,000 → Use NewsAPI Free + Guardian
│     ├─ 1,000-10,000 → Add RSS feeds
│     └─ > 10,000 → Premium APIs + scaling
│
└─ NO
   └─ Keep current manual system
```

### When to Start?
- ✅ **Now** - If you want to monitor news 24/7
- ✅ **Now** - If you need international coverage
- ✅ **Now** - If you want automated alerts
- ⏸️ **Wait** - If budget is a concern (start free!)
- ⏸️ **Wait** - If manual analysis is sufficient

---

## 🚀 Next Steps

### Option A: Full Implementation (Recommended)
1. Review full plan: `REALTIME_NEWS_API_PLAN.md`
2. Start with Phase 1 (API integration)
3. Follow 8-week timeline
4. Deploy to production

**Timeline:** 6-8 weeks  
**Investment:** 40-50 hours  
**Cost:** $0 (first 3 months)

### Option B: Quick Prototype (Fast Start)
1. Register for NewsAPI
2. Test basic fetching (today)
3. Connect to existing analyzer (tomorrow)
4. Evaluate results (this week)

**Timeline:** 2-3 days  
**Investment:** 4-6 hours  
**Cost:** $0

### Option C: Hybrid Approach (Flexible)
1. Start with manual + API testing
2. Build incrementally (1 feature/week)
3. Scale based on results

**Timeline:** Flexible  
**Investment:** As available  
**Cost:** $0 initially

---

## 📞 Resources

### Documentation
- **Full Plan:** `docs/REALTIME_NEWS_API_PLAN.md`
- **API Docs:** https://newsapi.org/docs
- **Guardian API:** https://open-platform.theguardian.com/documentation/

### Tools
- **NewsAPI Python:** `pip install newsapi-python`
- **Scheduler:** `pip install APScheduler`
- **Database:** `pip install SQLAlchemy`

### Support
- GitHub Issues: [Your repo issues]
- API Support: NewsAPI support page
- Community: Stack Overflow

---

## ✅ Checklist

### Before Starting
- [ ] Understand current system capabilities
- [ ] Define monitoring requirements
- [ ] Estimate budget and timeline
- [ ] Get stakeholder approval

### Setup Phase
- [ ] Register for NewsAPI
- [ ] Register for Guardian API
- [ ] Install dependencies
- [ ] Create `.env` configuration
- [ ] Test API connectivity

### Development Phase
- [ ] Build API client modules
- [ ] Implement news fetcher
- [ ] Create processing pipeline
- [ ] Setup task scheduler
- [ ] Add alert system

### Testing Phase
- [ ] Test API integrations
- [ ] Verify deduplication
- [ ] Validate analysis accuracy
- [ ] Test alert system
- [ ] Load testing

### Deployment Phase
- [ ] Deploy to cloud/server
- [ ] Setup monitoring
- [ ] Configure alerts
- [ ] Document operations
- [ ] Train users

---

**Ready to start?** Begin with:
```powershell
# 1. Get API key from NewsAPI.org
# 2. Test basic connection
pip install newsapi-python
python test_api.py

# 3. Review full implementation plan
# Open: docs/REALTIME_NEWS_API_PLAN.md
```

---

**Questions?** Check the full implementation plan or ask for specific guidance!
