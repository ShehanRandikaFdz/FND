/**
 * Authentication JavaScript for Fake News Detector
 * Handles login, registration, and authentication state
 */

// Initialize auth when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeAuth();
});

function initializeAuth() {
    console.log('🔐 Initializing authentication...');
    
    // Set up login form
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }
    
    // Set up register form
    const registerForm = document.getElementById('register-form');
    if (registerForm) {
        registerForm.addEventListener('submit', handleRegister);
    }
    
    console.log('✅ Auth initialized successfully!');
}

/**
 * Handle Login Form Submission
 */
async function handleLogin(event) {
    event.preventDefault();
    
    const loginBtn = document.getElementById('login-btn');
    const loginBtnText = document.getElementById('login-btn-text');
    const loginSpinner = document.getElementById('login-spinner');
    
    // Get form data
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value;
    
    // Validate inputs
    if (!email || !password) {
        showAlert('Please fill in all fields', 'error');
        return;
    }
    
    if (!isValidEmail(email)) {
        showAlert('Please enter a valid email address', 'error');
        return;
    }
    
    // Disable button and show loading
    loginBtn.disabled = true;
    loginBtnText.classList.add('hidden');
    loginSpinner.classList.remove('hidden');
    
    try {
        // Call login API
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                email: email,
                password: password
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Login successful
            showAlert('Login successful! Redirecting...', 'success');
            
            // Store user info in localStorage
            localStorage.setItem('userEmail', email);
            localStorage.setItem('userName', data.user.name);
            localStorage.setItem('userPlan', data.user.plan);
            
            // Redirect to home page after short delay
            setTimeout(() => {
                window.location.href = '/';
            }, 1500);
        } else {
            // Login failed
            showAlert(data.message || 'Login failed. Please check your credentials.', 'error');
            loginBtn.disabled = false;
            loginBtnText.classList.remove('hidden');
            loginSpinner.classList.add('hidden');
        }
    } catch (error) {
        console.error('Login error:', error);
        showAlert('An error occurred. Please try again.', 'error');
        loginBtn.disabled = false;
        loginBtnText.classList.remove('hidden');
        loginSpinner.classList.add('hidden');
    }
}

/**
 * Handle Register Form Submission
 */
async function handleRegister(event) {
    event.preventDefault();
    
    const registerBtn = document.getElementById('register-btn');
    const registerBtnText = document.getElementById('register-btn-text');
    const registerSpinner = document.getElementById('register-spinner');
    
    // Get form data
    const name = document.getElementById('name').value.trim();
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm-password').value;
    const plan = document.querySelector('input[name="plan"]:checked').value;
    const termsAccepted = document.getElementById('terms').checked;
    
    // Validate inputs
    if (!name || !email || !password || !confirmPassword) {
        showAlert('Please fill in all required fields', 'error');
        return;
    }
    
    if (!isValidEmail(email)) {
        showAlert('Please enter a valid email address', 'error');
        return;
    }
    
    if (password.length < 6) {
        showAlert('Password must be at least 6 characters long', 'error');
        return;
    }
    
    if (password !== confirmPassword) {
        showAlert('Passwords do not match', 'error');
        return;
    }
    
    if (!termsAccepted) {
        showAlert('Please accept the Terms of Service and Privacy Policy', 'error');
        return;
    }
    
    // Disable button and show loading
    registerBtn.disabled = true;
    registerBtnText.classList.add('hidden');
    registerSpinner.classList.remove('hidden');
    
    try {
        // Call register API
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: name,
                email: email,
                password: password,
                plan: plan
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Registration successful
            showAlert('Account created successfully! Redirecting...', 'success');
            
            // Store user info in localStorage
            localStorage.setItem('userEmail', email);
            localStorage.setItem('userName', name);
            localStorage.setItem('userPlan', plan);
            
            // Redirect to home page after short delay
            setTimeout(() => {
                window.location.href = '/';
            }, 1500);
        } else {
            // Registration failed
            showAlert(data.message || 'Registration failed. Please try again.', 'error');
            registerBtn.disabled = false;
            registerBtnText.classList.remove('hidden');
            registerSpinner.classList.add('hidden');
        }
    } catch (error) {
        console.error('Registration error:', error);
        showAlert('An error occurred. Please try again.', 'error');
        registerBtn.disabled = false;
        registerBtnText.classList.remove('hidden');
        registerSpinner.classList.add('hidden');
    }
}

/**
 * Show Alert Message
 */
function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alert-container');
    if (!alertContainer) return;
    
    // Clear existing alerts
    alertContainer.innerHTML = '';
    
    // Determine colors based on type
    let bgColor, textColor, icon;
    if (type === 'success') {
        bgColor = 'bg-green-100 dark:bg-green-900/30';
        textColor = 'text-green-800 dark:text-green-200';
        icon = '✅';
    } else if (type === 'error') {
        bgColor = 'bg-red-100 dark:bg-red-900/30';
        textColor = 'text-red-800 dark:text-red-200';
        icon = '❌';
    } else {
        bgColor = 'bg-blue-100 dark:bg-blue-900/30';
        textColor = 'text-blue-800 dark:text-blue-200';
        icon = 'ℹ️';
    }
    
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `${bgColor} ${textColor} px-4 py-3 rounded-lg flex items-center space-x-2 animate-fade-in`;
    alertDiv.innerHTML = `
        <span class="text-xl">${icon}</span>
        <span class="font-semibold">${message}</span>
    `;
    
    alertContainer.appendChild(alertDiv);
    alertContainer.classList.remove('hidden');
    
    // Auto-hide after 5 seconds for success messages
    if (type === 'success') {
        setTimeout(() => {
            alertContainer.classList.add('hidden');
        }, 5000);
    }
}

/**
 * Validate Email Format
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Handle Logout
 */
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
            
            // Redirect to home page
            window.location.href = '/';
        }
    } catch (error) {
        console.error('Logout error:', error);
    }
}

/**
 * Check if user is authenticated
 */
function isAuthenticated() {
    return localStorage.getItem('userEmail') !== null;
}

/**
 * Get current user info
 */
function getCurrentUser() {
    if (!isAuthenticated()) {
        return null;
    }
    
    return {
        email: localStorage.getItem('userEmail'),
        name: localStorage.getItem('userName'),
        plan: localStorage.getItem('userPlan')
    };
}

// Export functions for use in other scripts
window.authModule = {
    handleLogout,
    isAuthenticated,
    getCurrentUser
};
