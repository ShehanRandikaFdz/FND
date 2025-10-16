# API Documentation - Commercial Fake News Detector

## 🌐 API Overview

The Commercial Fake News Detector provides a comprehensive REST API for AI-powered content verification with subscription-based access control.

### Base URL
```
Production: https://api.factcheckpro.com
Staging: https://staging-api.factcheckpro.com
Development: http://localhost:5000
```

### Authentication
All API requests require authentication using API keys or session-based authentication.

```bash
# API Key Authentication
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.factcheckpro.com/analyze

# Session Authentication (for web applications)
curl -H "Cookie: session=YOUR_SESSION_ID" \
     -H "Content-Type: application/json" \
     https://api.factcheckpro.com/analyze
```

## 🔐 Authentication Endpoints

### POST /auth/register
Register a new user account.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "secure_password",
  "name": "John Doe",
  "plan": "starter"
}
```

**Response:**
```json
{
  "success": true,
  "message": "User registered successfully",
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "name": "John Doe",
    "plan": "starter",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

### POST /auth/login
Authenticate user and create session.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Login successful",
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "name": "John Doe",
    "plan": "starter"
  },
  "session_token": "session_token_123"
}
```

### POST /auth/logout
Logout user and invalidate session.

**Response:**
```json
{
  "message": "Logged out successfully"
}
```

## 📊 Subscription Management

### GET /subscriptions/plans
Get available subscription plans.

**Response:**
```json
{
  "plans": {
    "starter": {
      "name": "Starter",
      "price": 19.00,
      "analyses_limit": 500,
      "features": [
        "Basic text analysis",
        "Email support",
        "Standard confidence scores"
      ],
      "restrictions": {
        "url_analysis": false,
        "news_api_verification": false
      }
    },
    "professional": {
      "name": "Professional",
      "price": 99.00,
      "analyses_limit": 5000,
      "features": [
        "All Starter features",
        "URL analysis",
        "NewsAPI verification",
        "API access"
      ],
      "restrictions": {
        "url_analysis": true,
        "news_api_verification": true
      }
    },
    "business": {
      "name": "Business",
      "price": 299.00,
      "analyses_limit": 25000,
      "features": [
        "All Professional features",
        "Batch processing",
        "Custom integrations"
      ],
      "restrictions": {
        "url_analysis": true,
        "news_api_verification": true,
        "batch_processing": true
      }
    },
    "enterprise": {
      "name": "Enterprise",
      "price": 999.00,
      "analyses_limit": -1,
      "features": [
        "All Business features",
        "Unlimited analyses",
        "White-label options"
      ],
      "restrictions": {
        "url_analysis": true,
        "news_api_verification": true,
        "batch_processing": true
      }
    }
  }
}
```

### POST /subscriptions/upgrade
Upgrade user subscription plan.

**Request:**
```json
{
  "plan": "professional",
  "payment_method": "card_123"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Subscription upgraded successfully",
  "subscription": {
    "plan": "professional",
    "price": 99.00,
    "next_billing_date": "2024-02-01T00:00:00Z"
  }
}
```

### GET /subscriptions/usage
Get user usage statistics.

**Response:**
```json
{
  "success": true,
  "usage": {
    "monthly_usage": 150,
    "total_usage": 500,
    "limit": 500,
    "remaining": 350,
    "plan": "starter",
    "reset_date": "2024-02-01T00:00:00Z"
  }
}
```

## 🔍 Analysis Endpoints

### POST /analyze
Analyze text for fake news detection.

**Request:**
```json
{
  "text": "Breaking: Scientists discover new planet with life forms"
}
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 85.5,
  "news_api_results": {
    "found": false,
    "articles": [],
    "error": "NewsAPI verification requires Professional plan or higher"
  },
  "individual_results": {
    "svm": {
      "model_name": "SVM",
      "prediction": "FAKE",
      "confidence": 82.3,
      "probability_fake": 82.3,
      "probability_true": 17.7
    },
    "lstm": {
      "model_name": "LSTM",
      "prediction": "FAKE",
      "confidence": 88.7,
      "probability_fake": 88.7,
      "probability_true": 11.3
    },
    "bert": {
      "model_name": "BERT",
      "prediction": "FAKE",
      "confidence": 85.5,
      "probability_fake": 85.5,
      "probability_true": 14.5
    }
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "text": "Breaking: Scientists discover new planet with life forms",
  "explanation": "❌ ANALYSIS RESULT: High probability of misinformation (confidence: 85.5%). Text patterns suggest sensationalism or lack of factual basis. No matching articles found in trusted sources.",
  "commercial_info": {
    "user_plan": "starter",
    "authenticated": true,
    "feature_restrictions": {
      "news_api_verification": true
    }
  }
}
```

### POST /analyze-url
Analyze news article from URL (Professional+ plans only).

**Request:**
```json
{
  "url": "https://example.com/news-article"
}
```

**Response:**
```json
{
  "prediction": "TRUE",
  "confidence": 92.1,
  "news_api_results": {
    "found": true,
    "articles": [
      {
        "title": "Scientists discover new planet with life forms",
        "source": "BBC News",
        "publishedAt": "2024-01-01T10:00:00Z",
        "url": "https://bbc.com/news/science",
        "similarity_score": 0.95
      }
    ],
    "best_match": {
      "title": "Scientists discover new planet with life forms",
      "source": "BBC News",
      "publishedAt": "2024-01-01T10:00:00Z",
      "url": "https://bbc.com/news/science",
      "similarity_score": 0.95
    }
  },
  "individual_results": {
    "svm": {
      "model_name": "SVM",
      "prediction": "TRUE",
      "confidence": 89.2,
      "probability_fake": 10.8,
      "probability_true": 89.2
    },
    "lstm": {
      "model_name": "LSTM",
      "prediction": "TRUE",
      "confidence": 94.5,
      "probability_fake": 5.5,
      "probability_true": 94.5
    },
    "bert": {
      "model_name": "BERT",
      "prediction": "TRUE",
      "confidence": 92.1,
      "probability_fake": 7.9,
      "probability_true": 92.1
    }
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "url": "https://example.com/news-article",
  "article_title": "Scientists discover new planet with life forms",
  "article_source": "Example News",
  "article_text": "Scientists have discovered a new planet with potential life forms...",
  "explanation": "✅ ANALYSIS RESULT: High probability of credible content (confidence: 92.1%). Text patterns are consistent with factual reporting. Found 1 matching article from trusted sources.",
  "commercial_info": {
    "user_plan": "professional",
    "feature_restrictions": {
      "news_api_verification": false
    }
  }
}
```

### POST /analyze-batch
Analyze multiple texts in batch (Business+ plans only).

**Request:**
```json
{
  "texts": [
    "Breaking: Scientists discover new planet with life forms",
    "Local weather forecast for tomorrow",
    "Stock market reaches new all-time high"
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Breaking: Scientists discover new planet with life forms",
      "prediction": "FAKE",
      "confidence": 85.5,
      "explanation": "❌ ANALYSIS RESULT: High probability of misinformation (confidence: 85.5%)."
    },
    {
      "text": "Local weather forecast for tomorrow",
      "prediction": "TRUE",
      "confidence": 78.2,
      "explanation": "✅ ANALYSIS RESULT: High probability of credible content (confidence: 78.2%)."
    },
    {
      "text": "Stock market reaches new all-time high",
      "prediction": "TRUE",
      "confidence": 91.3,
      "explanation": "✅ ANALYSIS RESULT: High probability of credible content (confidence: 91.3%)."
    }
  ],
  "batch_id": "batch_123",
  "total_processed": 3,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 📰 News Endpoints

### POST /news/fetch
Fetch latest news articles (authenticated users only).

**Request:**
```json
{
  "country": "us",
  "category": "general",
  "page_size": 10
}
```

**Response:**
```json
{
  "articles": [
    {
      "title": "Scientists discover new planet with life forms",
      "description": "A team of scientists has discovered a new planet with potential life forms...",
      "url": "https://bbc.com/news/science",
      "source": "BBC News",
      "published_at": "2024-01-01T10:00:00Z",
      "credibility_score": 0.95,
      "prediction": "TRUE",
      "confidence": 92.1
    }
  ],
  "total_results": 100,
  "page": 1,
  "page_size": 10
}
```

### GET /news/sources
Get available news sources.

**Response:**
```json
{
  "sources": [
    {
      "id": "bbc-news",
      "name": "BBC News",
      "description": "BBC News is the operational business division of the British Broadcasting Corporation",
      "url": "https://www.bbc.com/news",
      "category": "general",
      "language": "en",
      "country": "gb"
    }
  ]
}
```

## 📊 Analytics Endpoints

### GET /analytics/usage
Get detailed usage analytics.

**Response:**
```json
{
  "success": true,
  "analytics": {
    "total_analyses": 1500,
    "monthly_analyses": 150,
    "daily_analyses": 5,
    "analyses_by_type": {
      "text": 1200,
      "url": 250,
      "batch": 50
    },
    "analyses_by_plan": {
      "starter": 1000,
      "professional": 400,
      "business": 100
    },
    "usage_trends": {
      "daily": [
        {"date": "2024-01-01", "count": 5},
        {"date": "2024-01-02", "count": 8},
        {"date": "2024-01-03", "count": 12}
      ],
      "monthly": [
        {"month": "2024-01", "count": 150},
        {"month": "2024-02", "count": 200},
        {"month": "2024-03", "count": 250}
      ]
    }
  }
}
```

### GET /analytics/performance
Get system performance metrics.

**Response:**
```json
{
  "success": true,
  "performance": {
    "response_time": {
      "average": 1.2,
      "p95": 2.5,
      "p99": 5.0
    },
    "throughput": {
      "requests_per_second": 100,
      "analyses_per_minute": 50
    },
    "accuracy": {
      "overall": 0.92,
      "by_model": {
        "svm": 0.89,
        "lstm": 0.91,
        "bert": 0.94
      }
    },
    "uptime": {
      "current": 99.9,
      "last_30_days": 99.8
    }
  }
}
```

## 🔧 Admin Endpoints

### GET /admin/users
Get all users (admin only).

**Response:**
```json
{
  "success": true,
  "users": [
    {
      "id": "user_123",
      "email": "user@example.com",
      "name": "John Doe",
      "plan": "starter",
      "created_at": "2024-01-01T00:00:00Z",
      "last_login": "2024-01-15T12:00:00Z",
      "is_active": true
    }
  ],
  "total": 1000,
  "page": 1,
  "page_size": 50
}
```

### POST /admin/users/{user_id}/suspend
Suspend user account (admin only).

**Response:**
```json
{
  "success": true,
  "message": "User account suspended successfully"
}
```

### GET /admin/analytics/revenue
Get revenue analytics (admin only).

**Response:**
```json
{
  "success": true,
  "revenue": {
    "total_revenue": 50000,
    "monthly_revenue": 5000,
    "revenue_by_plan": {
      "starter": 10000,
      "professional": 25000,
      "business": 12000,
      "enterprise": 3000
    },
    "revenue_trends": {
      "daily": [
        {"date": "2024-01-01", "revenue": 200},
        {"date": "2024-01-02", "revenue": 250},
        {"date": "2024-01-03", "revenue": 300}
      ]
    }
  }
}
```

## 🚨 Error Handling

### Error Response Format
```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": "Additional error details",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common Error Codes
- `AUTHENTICATION_REQUIRED`: Authentication required
- `INVALID_CREDENTIALS`: Invalid email or password
- `USAGE_LIMIT_EXCEEDED`: Usage limit exceeded
- `PLAN_UPGRADE_REQUIRED`: Plan upgrade required
- `INVALID_REQUEST`: Invalid request format
- `RATE_LIMIT_EXCEEDED`: Rate limit exceeded
- `SERVER_ERROR`: Internal server error

### HTTP Status Codes
- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Access denied
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## 📝 Rate Limiting

### Rate Limits by Plan
- **Starter**: 100 requests/hour
- **Professional**: 1000 requests/hour
- **Business**: 5000 requests/hour
- **Enterprise**: 10000 requests/hour

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## 🔒 Security

### API Key Security
- API keys are required for all requests
- Keys are tied to user accounts and plans
- Keys can be regenerated if compromised
- Keys expire after 1 year of inactivity

### Data Privacy
- All text analysis is processed securely
- No personal data is stored with analysis results
- All data transmission is encrypted (HTTPS)
- GDPR compliance for EU users

### CORS Configuration
```javascript
// Allowed origins
https://app.factcheckpro.com
https://staging.factcheckpro.com
http://localhost:3000

// Allowed methods
GET, POST, PUT, DELETE, OPTIONS

// Allowed headers
Authorization, Content-Type, X-Requested-With
```

## 📚 SDKs and Libraries

### Python SDK
```python
from factcheckpro import FactCheckPro

client = FactCheckPro(api_key="your_api_key")

# Analyze text
result = client.analyze("Breaking news: Scientists discover new planet")
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence}")

# Analyze URL
result = client.analyze_url("https://example.com/news-article")
print(f"Prediction: {result.prediction}")
```

### JavaScript SDK
```javascript
import FactCheckPro from 'factcheckpro';

const client = new FactCheckPro('your_api_key');

// Analyze text
const result = await client.analyze('Breaking news: Scientists discover new planet');
console.log(`Prediction: ${result.prediction}`);
console.log(`Confidence: ${result.confidence}`);
```

### cURL Examples
```bash
# Analyze text
curl -X POST https://api.factcheckpro.com/analyze \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news: Scientists discover new planet"}'

# Analyze URL
curl -X POST https://api.factcheckpro.com/analyze-url \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/news-article"}'
```

## 🎯 Conclusion

This API documentation provides comprehensive information about the Commercial Fake News Detector API, including:

- **Authentication**: User registration, login, and session management
- **Subscription Management**: Plan management and usage tracking
- **Analysis Endpoints**: Text, URL, and batch analysis
- **News Endpoints**: News fetching and source management
- **Analytics**: Usage and performance metrics
- **Admin Functions**: User and system management
- **Error Handling**: Comprehensive error codes and responses
- **Security**: API key management and data privacy
- **SDKs**: Python and JavaScript SDKs for easy integration

The API is designed to be developer-friendly with clear documentation, consistent response formats, and comprehensive error handling.
