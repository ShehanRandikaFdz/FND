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
    
    // Check authentication state
    updateAuthUI();
    
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
    
    // Clear history button
    const clearBtn = document.getElementById('clear-history-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearHistory);
    }
    
    // Analyze URL button
    const analyzeUrlBtn = document.getElementById('analyze-url-btn');
    if (analyzeUrlBtn) {
        analyzeUrlBtn.addEventListener('click', analyzeUrl);
    }
    
    // Logout buttons
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }
    
    const logoutBtnNav = document.getElementById('logout-btn-nav');
    if (logoutBtnNav) {
        logoutBtnNav.addEventListener('click', handleLogout);
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
        // Remove any existing event listeners to prevent duplicates
        const newToggle = toggle.cloneNode(true);
        toggle.parentNode.replaceChild(newToggle, toggle);
        
        newToggle.addEventListener('click', function() {
            const html = document.documentElement;
            const isDark = html.classList.contains('dark');
            
            if (isDark) {
                html.classList.remove('dark');
                newToggle.textContent = '🌙';
                newToggle.innerHTML = '<span class="text-xl">🌙</span>';
                localStorage.setItem('darkMode', 'false');
            } else {
                html.classList.add('dark');
                newToggle.textContent = '☀️';
                newToggle.innerHTML = '<span class="text-xl">☀️</span>';
                localStorage.setItem('darkMode', 'true');
            }
        });
        
        // Load saved preference
        const savedMode = localStorage.getItem('darkMode');
        if (savedMode === 'true') {
            document.documentElement.classList.add('dark');
            newToggle.innerHTML = '<span class="text-xl">☀️</span>';
        } else {
            document.documentElement.classList.remove('dark');
            newToggle.innerHTML = '<span class="text-xl">🌙</span>';
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

async function analyzeUrl() {
    const urlInput = document.getElementById('news-url');
    const url = urlInput.value.trim();
    
    if (!url) {
        showError('Please enter a URL to analyze.');
        return;
    }
    
    // Basic URL validation
    try {
        new URL(url);
    } catch (e) {
        showError('Please enter a valid URL.');
        return;
    }
    
    if (isAnalyzing) {
        return; // Prevent multiple simultaneous requests
    }
    
    isAnalyzing = true;
    updateAnalyzingUI(true);
    
    try {
        console.log('🔗 Analyzing URL...');
        
        const response = await fetch('/analyze-url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ URL analysis complete:', result);
        
        // Update UI with results
        updateUI(result);
        updateHistory();
        
    } catch (error) {
        console.error('❌ Error analyzing URL:', error);
        showError('Failed to analyze URL. Please try again.');
    } finally {
        isAnalyzing = false;
        updateAnalyzingUI(false);
    }
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
    
    // Update URL-specific info (if analyzing URL)
    if (result.url) {
        updateUrlInfo(result);
    }
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
    if (result.news_api_results && result.news_api_results.found_online && result.news_api_results.all_matches) {
        const articles = result.news_api_results.all_matches;
        const bestMatch = result.news_api_results.best_match;
        
        html += '<div class="mb-6">';
        html += '<div class="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">';
        html += '<h4 class="font-semibold text-green-700 dark:text-green-300 mb-3 flex items-center">';
        html += '<span class="mr-2">✅</span> Online Verification Found';
        html += '</h4>';
        
        // Best match article
        if (bestMatch) {
            const similarity = Math.round((bestMatch.similarity_score || 0) * 100);
            const publishedDate = bestMatch.publishedAt ? new Date(bestMatch.publishedAt).toLocaleDateString() : 'Unknown Date';
            
            html += '<div class="bg-white dark:bg-gray-800 rounded-lg p-4 mb-3 border border-green-200 dark:border-green-700">';
            html += '<div class="flex items-start justify-between mb-2">';
            html += '<h5 class="font-semibold text-gray-900 dark:text-gray-100 text-sm leading-tight">📰 Best Match</h5>';
            html += `<span class="bg-green-100 dark:bg-green-800 text-green-800 dark:text-green-200 px-2 py-1 rounded-full text-xs font-medium">${similarity}% Similar</span>`;
            html += '</div>';
            html += `<p class="font-medium text-gray-800 dark:text-gray-200 mb-2">${bestMatch.title || 'Unknown Title'}</p>`;
            html += '<div class="text-xs text-gray-600 dark:text-gray-400 space-y-1">';
            html += `<p><strong>🏢 Source:</strong> ${bestMatch.source?.name || 'Unknown Source'}</p>`;
            html += `<p><strong>📅 Published:</strong> ${publishedDate}</p>`;
            if (bestMatch.url) {
                html += `<p><strong>🔗 Link:</strong> <a href="${bestMatch.url}" target="_blank" class="text-blue-600 dark:text-blue-400 hover:underline">Read Original Article →</a></p>`;
            }
            html += '</div>';
            html += '</div>';
        }
        
        // Additional articles
        if (articles.length > 1) {
            html += '<div class="mt-3">';
            html += `<h6 class="font-medium text-gray-700 dark:text-gray-300 mb-2">📚 Other Matches (${articles.length - 1} more):</h6>`;
            html += '<div class="space-y-2">';
            
            articles.slice(1, 4).forEach((article, index) => {
                const similarity = Math.round((article.similarity_score || 0) * 100);
                html += `
                    <div class="bg-gray-50 dark:bg-gray-700 rounded p-3 border-l-4 border-blue-400">
                        <p class="font-medium text-sm text-gray-800 dark:text-gray-200 mb-1">${article.title || 'Unknown Title'}</p>
                        <div class="text-xs text-gray-600 dark:text-gray-400">
                            <span class="font-medium">${article.source?.name || 'Unknown Source'}</span>
                            <span class="mx-2">•</span>
                            <span>${similarity}% similar</span>
                            ${article.url ? `<span class="mx-2">•</span><a href="${article.url}" target="_blank" class="text-blue-600 dark:text-blue-400 hover:underline">Read →</a>` : ''}
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            html += '</div>';
        }
        
        html += '</div>';
        html += '</div>';
    } else {
        // No online sources found - silently skip the warning
        // Removed the warning message as requested
    }
    
    // Add ML explanation (formatted) - This is the main fix
    if (result.explanation) {
        console.log('🤖 AI Explanation found:', result.explanation);
        
        // Format the explanation with proper line breaks and styling
        const formattedExplanation = result.explanation
            .replace(/\n/g, '<br>')
            .replace(/✅/g, '<span class="text-green-600 dark:text-green-400">✅</span>')
            .replace(/❌/g, '<span class="text-red-600 dark:text-red-400">❌</span>')
            .replace(/⚠️/g, '<span class="text-yellow-600 dark:text-yellow-400">⚠️</span>')
            .replace(/📰/g, '<span class="text-blue-600 dark:text-blue-400">📰</span>')
            .replace(/🏢/g, '<span class="text-purple-600 dark:text-purple-400">🏢</span>')
            .replace(/📅/g, '<span class="text-indigo-600 dark:text-indigo-400">📅</span>')
            .replace(/🎯/g, '<span class="text-pink-600 dark:text-pink-400">🎯</span>')
            .replace(/🔗/g, '<span class="text-cyan-600 dark:text-cyan-400">🔗</span>')
            .replace(/📚/g, '<span class="text-orange-600 dark:text-orange-400">📚</span>')
            .replace(/🤖/g, '<span class="text-blue-600 dark:text-blue-400">🤖</span>')
            .replace(/📊/g, '<span class="text-purple-600 dark:text-purple-400">📊</span>');
        
        html += '<div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">';
        html += '<h5 class="font-semibold text-gray-700 dark:text-gray-300 mb-2">🤖 AI Analysis Result:</h5>';
        html += `<div class="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">${formattedExplanation}</div>`;
        html += '</div>';
    } else {
        console.log('⚠️ No AI explanation found in result:', result);
        
        // Fallback explanation
        const prediction = result.prediction;
        const confidence = Math.round(result.confidence);
        
        html += '<div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">';
        html += '<h5 class="font-semibold text-gray-700 dark:text-gray-300 mb-2">🤖 AI Analysis Result:</h5>';
        
        if (prediction === 'FAKE') {
            html += `<p class="text-sm text-red-600 dark:text-red-400">❌ High probability of misinformation (confidence: ${confidence}%). Patterns detected are commonly found in fabricated or satirical content.</p>`;
        } else if (prediction === 'TRUE') {
            html += `<p class="text-sm text-green-600 dark:text-green-400">✅ High probability of credible content (confidence: ${confidence}%). Text patterns are consistent with factual reporting.</p>`;
        } else {
            html += `<p class="text-sm text-yellow-600 dark:text-yellow-400">⚠️ Analysis is inconclusive (confidence: ${confidence}%). Additional verification may be needed.</p>`;
        }
        
        html += '</div>';
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

function updateUrlInfo(result) {
    // Update source info with URL analysis results
    const sourceEl = document.getElementById('source');
    const publishedEl = document.getElementById('published');
    const urlEl = document.getElementById('url');
    
    if (sourceEl && result.article_source) {
        sourceEl.textContent = result.article_source;
    }
    
    if (publishedEl) {
        publishedEl.textContent = new Date().toLocaleDateString();
    }
    
    if (urlEl && result.url) {
        urlEl.textContent = 'View Original';
        urlEl.href = result.url;
        urlEl.target = '_blank';
    }
    
    // Show article title if available
    if (result.article_title && result.article_title !== 'Unknown Title') {
        // You could add a title display element here if needed
        console.log('📰 Article Title:', result.article_title);
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

/**
 * Authentication Functions
 */

// Update UI based on authentication state
function updateAuthUI() {
    const user = getCurrentUser();
    const authButtons = document.getElementById('auth-buttons');
    const userMenu = document.getElementById('user-menu');
    const profileSection = document.getElementById('profile-section');
    
    if (user) {
        // User is logged in
        if (authButtons) authButtons.classList.add('hidden');
        if (userMenu) userMenu.classList.remove('hidden');
        if (profileSection) profileSection.classList.remove('hidden');
        
        // Update user info in navigation
        const userInitialNav = document.getElementById('user-initial-nav');
        const usernameNav = document.getElementById('username-nav');
        const userPlanNav = document.getElementById('user-plan-nav');
        
        if (userInitialNav && user.name) {
            userInitialNav.textContent = user.name.charAt(0).toUpperCase();
        }
        if (usernameNav && user.name) {
            usernameNav.textContent = user.name;
        }
        if (userPlanNav && user.plan) {
            userPlanNav.textContent = formatPlanName(user.plan);
        }
        
        // Update user info in profile section
        const userInitial = document.getElementById('user-initial');
        const username = document.getElementById('username');
        const userPlan = document.getElementById('user-plan');
        
        if (userInitial && user.name) {
            userInitial.textContent = user.name.charAt(0).toUpperCase();
        }
        if (username && user.name) {
            username.textContent = user.name;
        }
        if (userPlan && user.plan) {
            userPlan.textContent = formatPlanName(user.plan);
        }
        
        // Load usage stats
        loadUsageStats();
    } else {
        // User is not logged in
        if (authButtons) authButtons.classList.remove('hidden');
        if (userMenu) userMenu.classList.add('hidden');
        if (profileSection) profileSection.classList.add('hidden');
    }
}

// Get current user from localStorage
function getCurrentUser() {
    const email = localStorage.getItem('userEmail');
    const name = localStorage.getItem('userName');
    const plan = localStorage.getItem('userPlan');
    
    if (email && name && plan) {
        return { email, name, plan };
    }
    return null;
}

// Format plan name for display
function formatPlanName(plan) {
    const planNames = {
        'starter': 'Starter Plan',
        'professional': 'Professional Plan',
        'business': 'Business Plan',
        'enterprise': 'Enterprise Plan'
    };
    return planNames[plan] || plan;
}

// Load usage statistics
async function loadUsageStats() {
    try {
        const response = await fetch('/api/user/usage');
        if (response.ok) {
            const data = await response.json();
            const usageStats = document.getElementById('usage-stats');
            
            if (usageStats && data.usage && data.plan) {
                const used = data.usage.monthly_usage || 0;
                const limit = data.plan.analyses_limit || 0;
                
                if (limit === -1) {
                    usageStats.textContent = `${used} analyses (Unlimited)`;
                } else {
                    usageStats.textContent = `${used} / ${limit.toLocaleString()} analyses`;
                }
            }
        }
    } catch (error) {
        console.error('Error loading usage stats:', error);
    }
}

// Handle logout
async function handleLogout() {
    try {
        const response = await fetch('/api/auth/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (response.ok) {
            // Clear localStorage
            localStorage.removeItem('userEmail');
            localStorage.removeItem('userName');
            localStorage.removeItem('userPlan');
            
            // Reload page to update UI
            window.location.reload();
        }
    } catch (error) {
        console.error('Logout error:', error);
    }
}

function truncateText(text, maxLength = 100) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}
