/**
 * Main JavaScript for Fake News Detector
 * Handles API calls, UI updates, and history management
 */

// Global variables
let isAnalyzing = false;
let currentHistory = [];

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    console.log('🚀 Initializing Fake News Detector...');
    
    // Set up event listeners
    setupEventListeners();
    
    // Load history
    loadHistory();
    
    // Set up dark mode toggle
    setupDarkModeToggle();
    
    console.log('✅ App initialized successfully!');
}

function setupEventListeners() {
    // Check credibility button
    const checkBtn = document.getElementById('check-credibility-btn');
    if (checkBtn) {
        checkBtn.addEventListener('click', checkCredibility);
    }
    
    // Fetch news button
    const fetchBtn = document.getElementById('fetch-news-btn');
    if (fetchBtn) {
        fetchBtn.addEventListener('click', fetchLatestNews);
    }
    
    // Clear history button
    const clearBtn = document.getElementById('clear-history-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearHistory);
    }
    
    // Enter key in textarea
    const textarea = document.getElementById('news-text');
    if (textarea) {
        textarea.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                checkCredibility();
            }
        });
    }
}

function setupDarkModeToggle() {
    const toggle = document.getElementById('dark-mode-toggle');
    if (toggle) {
        toggle.addEventListener('click', function() {
            const html = document.documentElement;
            const isDark = html.classList.contains('dark');
            
            if (isDark) {
                html.classList.remove('dark');
                toggle.textContent = '🌙 Dark';
                localStorage.setItem('darkMode', 'false');
            } else {
                html.classList.add('dark');
                toggle.textContent = '☀️ Light';
                localStorage.setItem('darkMode', 'true');
            }
        });
        
        // Load saved preference
        const savedMode = localStorage.getItem('darkMode');
        if (savedMode === 'true') {
            document.documentElement.classList.add('dark');
            toggle.textContent = '☀️ Light';
        }
    }
}

async function checkCredibility() {
    const textarea = document.getElementById('news-text');
    const text = textarea.value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }
    
    if (isAnalyzing) {
        return; // Prevent multiple simultaneous requests
    }
    
    isAnalyzing = true;
    updateAnalyzingUI(true);
    
    try {
        console.log('🔍 Analyzing text...');
        
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ Analysis complete:', result);
        
        updateUI(result);
        updateHistory();
        
    } catch (error) {
        console.error('❌ Error analyzing text:', error);
        showError('Failed to analyze text. Please try again.');
    } finally {
        isAnalyzing = false;
        updateAnalyzingUI(false);
    }
}

async function fetchLatestNews() {
    const fetchBtn = document.getElementById('fetch-news-btn');
    const originalText = fetchBtn.textContent;
    
    try {
        fetchBtn.textContent = '📡 Fetching...';
        fetchBtn.disabled = true;
        
        console.log('📰 Fetching latest news...');
        
        const response = await fetch('/fetch-news', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                country: 'us',
                category: 'general',
                page_size: 5
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ News fetched:', data);
        
        displayLatestNews(data.articles || []);
        
    } catch (error) {
        console.error('❌ Error fetching news:', error);
        showError('Failed to fetch latest news. Please try again.');
    } finally {
        fetchBtn.textContent = originalText;
        fetchBtn.disabled = false;
    }
}

function displayLatestNews(articles) {
    if (!articles || articles.length === 0) {
        showError('No news articles found.');
        return;
    }
    
    // Display first article in textarea
    const firstArticle = articles[0];
    const textarea = document.getElementById('news-text');
    
    let newsText = firstArticle.title || '';
    if (firstArticle.description) {
        newsText += '\n\n' + firstArticle.description;
    }
    if (firstArticle.content) {
        newsText += '\n\n' + firstArticle.content;
    }
    
    textarea.value = newsText;
    
    // Show success message
    showSuccess(`Fetched ${articles.length} news articles. First article loaded for analysis.`);
}

function updateUI(result) {
    console.log('🎨 Updating UI with result:', result);
    
    // Update status
    updateStatus(result.prediction, result.confidence);
    
    // Update confidence score
    updateConfidence(result.confidence);
    
    // Update explanation
    updateExplanation(result);
    
    // Update source info (if available)
    updateSourceInfo(result);
}

function updateStatus(prediction, confidence) {
    const statusEl = document.getElementById('status');
    if (!statusEl) return;
    
    let statusClass, statusText, statusIcon;
    
    if (prediction === 'FAKE') {
        statusClass = 'bg-red-500/20 text-red-500';
        statusText = '🔴 Fake News';
        statusIcon = '🔴';
    } else if (prediction === 'TRUE') {
        statusClass = 'bg-green-500/20 text-green-500';
        statusText = '🟢 Real News';
        statusIcon = '🟢';
    } else {
        statusClass = 'bg-yellow-500/20 text-yellow-500';
        statusText = '⚠️ Suspicious';
        statusIcon = '⚠️';
    }
    
    statusEl.className = `px-3 py-1 rounded-full text-sm font-medium ${statusClass}`;
    statusEl.textContent = statusText;
}

function updateConfidence(confidence) {
    const confidenceEl = document.getElementById('confidence');
    const confidenceBarEl = document.getElementById('confidence-bar');
    
    if (confidenceEl) {
        confidenceEl.textContent = `${Math.round(confidence)}%`;
    }
    
    if (confidenceBarEl) {
        confidenceBarEl.style.width = `${Math.round(confidence)}%`;
    }
}

function updateExplanation(result) {
    const explanationEl = document.getElementById('explanation-content');
    if (!explanationEl) return;
    
    let html = '';
    
    // Check if NewsAPI found articles
    if (result.news_api_results && result.news_api_results.found && result.news_api_results.articles) {
        const articles = result.news_api_results.articles;
        
        html += '<div class="mb-4">';
        html += '<h4 class="font-semibold text-green-600 dark:text-green-400 mb-2">✅ Found in Online Sources:</h4>';
        html += '<div class="space-y-3">';
        
        articles.slice(0, 3).forEach(article => {
            const similarity = Math.round((article.similarity_score || 0) * 100);
            html += `
                <div class="border-l-4 border-primary pl-3 py-2">
                    <p class="font-semibold text-sm">${article.title || 'Unknown Title'}</p>
                    <p class="text-xs text-gray-500 dark:text-gray-400">
                        <strong>Source:</strong> ${article.source || 'Unknown'} | 
                        <strong>Similarity:</strong> ${similarity}%
                    </p>
                    ${article.url ? `<a href="${article.url}" target="_blank" class="text-primary hover:underline text-xs">Read original →</a>` : ''}
                </div>
            `;
        });
        
        html += '</div>';
        html += '</div>';
    } else {
        // Show warning if no online sources found
        html += `
            <div class="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded mb-4">
                <p class="text-yellow-800 dark:text-yellow-200 text-sm">
                    ⚠️ No matching articles found in trusted online sources.
                </p>
            </div>
        `;
    }
    
    // Add ML explanation
    if (result.explanation) {
        html += `<p class="text-sm">${result.explanation}</p>`;
    } else {
        const prediction = result.prediction;
        const confidence = Math.round(result.confidence);
        
        if (prediction === 'FAKE') {
            html += `<p class="text-sm">Analysis indicates high probability of misinformation (confidence: ${confidence}%). Patterns detected are commonly found in fabricated or satirical content.</p>`;
        } else if (prediction === 'TRUE') {
            html += `<p class="text-sm">Analysis indicates high probability of credible content (confidence: ${confidence}%). Text patterns are consistent with factual reporting.</p>`;
        } else {
            html += `<p class="text-sm">Analysis is inconclusive (confidence: ${confidence}%). Additional verification may be needed.</p>`;
        }
    }
    
    explanationEl.innerHTML = html;
}

function updateSourceInfo(result) {
    // This would be populated if we had source information
    // For now, we'll show generic info
    const sourceEl = document.getElementById('source');
    const publishedEl = document.getElementById('published');
    const urlEl = document.getElementById('url');
    
    if (sourceEl) sourceEl.textContent = 'User Input';
    if (publishedEl) publishedEl.textContent = new Date().toLocaleDateString();
    if (urlEl) {
        urlEl.textContent = 'N/A';
        urlEl.href = '#';
    }
}

function updateAnalyzingUI(analyzing) {
    const checkBtn = document.getElementById('check-credibility-btn');
    const statusEl = document.getElementById('status');
    
    if (analyzing) {
        if (checkBtn) {
            checkBtn.textContent = '🔄 Analyzing...';
            checkBtn.disabled = true;
        }
        if (statusEl) {
            statusEl.textContent = '⏳ Analyzing...';
            statusEl.className = 'px-3 py-1 rounded-full text-sm font-medium bg-gray-500/20 text-gray-500';
        }
    } else {
        if (checkBtn) {
            checkBtn.textContent = 'Check Credibility';
            checkBtn.disabled = false;
        }
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/history');
        if (response.ok) {
            const data = await response.json();
            currentHistory = data.history || [];
            displayHistory();
        }
    } catch (error) {
        console.error('❌ Error loading history:', error);
    }
}

function updateHistory() {
    // Update the history display
    displayHistory();
}

function displayHistory() {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;
    
    if (currentHistory.length === 0) {
        historyList.innerHTML = `
            <div class="text-center text-gray-500 dark:text-gray-400 py-8">
                <p>No analysis history yet.</p>
                <p class="text-sm mt-2">Start by analyzing some text above!</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    currentHistory.slice(-10).reverse().forEach((item, index) => {
        const prediction = item.prediction;
        const confidence = Math.round(item.confidence);
        const timestamp = new Date(item.timestamp).toLocaleString();
        const text = item.text || 'Unknown text';
        
        let statusClass, statusText, statusIcon;
        
        if (prediction === 'FAKE') {
            statusClass = 'bg-red-500/20 text-red-500';
            statusText = '🔴 Fake';
            statusIcon = '🔴';
        } else if (prediction === 'TRUE') {
            statusClass = 'bg-green-500/20 text-green-500';
            statusText = '🟢 Real';
            statusIcon = '🟢';
        } else {
            statusClass = 'bg-yellow-500/20 text-yellow-500';
            statusText = '⚠️ Suspicious';
            statusIcon = '⚠️';
        }
        
        html += `
            <div class="p-4 rounded-md border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-background-dark/20 transition cursor-pointer" onclick="loadFromHistory('${text.replace(/'/g, "\\'")}')">
                <div class="flex justify-between items-start">
                    <p class="font-semibold text-gray-800 dark:text-gray-200 text-sm">${text.length > 60 ? text.substring(0, 60) + '...' : text}</p>
                    <span class="px-2 py-0.5 rounded-full text-xs font-medium ${statusClass} ml-2 whitespace-nowrap">${statusText}</span>
                </div>
                <div class="text-sm text-gray-500 dark:text-gray-400 mt-2 flex justify-between items-center">
                    <span>Confidence: ${confidence}%</span>
                    <span>${timestamp}</span>
                </div>
            </div>
        `;
    });
    
    historyList.innerHTML = html;
}

function loadFromHistory(text) {
    const textarea = document.getElementById('news-text');
    if (textarea) {
        textarea.value = text;
        textarea.scrollTop = 0;
    }
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear all analysis history?')) {
        return;
    }
    
    try {
        const response = await fetch('/clear-history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (response.ok) {
            currentHistory = [];
            displayHistory();
            showSuccess('History cleared successfully.');
        } else {
            throw new Error('Failed to clear history');
        }
    } catch (error) {
        console.error('❌ Error clearing history:', error);
        showError('Failed to clear history. Please try again.');
    }
}

function showError(message) {
    // Simple error display - could be enhanced with toast notifications
    console.error('❌ Error:', message);
    alert('Error: ' + message);
}

function showSuccess(message) {
    // Simple success display - could be enhanced with toast notifications
    console.log('✅ Success:', message);
    // Could add a toast notification here
}

// Utility functions
function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function truncateText(text, maxLength = 100) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}
