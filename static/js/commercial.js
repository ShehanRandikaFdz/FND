// Commercial JavaScript for handling login, registration, and payment forms

// Handle login form
async function handleLogin(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                email: formData.get('email'),
                password: formData.get('password')
            })
        });
        
        const result = await response.json();
        if (result.success) {
            window.location.href = '/dashboard';
        } else {
            alert(result.message);
        }
    } catch (error) {
        alert('Login failed. Please try again.');
        console.error('Login error:', error);
    }
}

// Handle registration form
async function handleRegister(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    
    try {
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                name: formData.get('name'),
                email: formData.get('email'),
                password: formData.get('password'),
                plan: formData.get('plan') || 'starter'
            })
        });
        
        const result = await response.json();
        if (result.success) {
            window.location.href = '/dashboard';
        } else {
            alert(result.message);
        }
    } catch (error) {
        alert('Registration failed. Please try again.');
        console.error('Registration error:', error);
    }
}

// Handle payment form
async function handlePayment(event) {
    event.preventDefault();
    const urlParams = new URLSearchParams(window.location.search);
    const plan = urlParams.get('plan') || 'professional';
    
    try {
        const response = await fetch('/api/payment/process', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({plan: plan})
        });
        
        const result = await response.json();
        if (result.success) {
            alert('Payment successful! Your plan has been upgraded.');
            window.location.href = '/dashboard';
        } else {
            alert(result.message);
        }
    } catch (error) {
        alert('Payment failed. Please try again.');
        console.error('Payment error:', error);
    }
}

// Handle logout
async function handleLogout() {
    try {
        const response = await fetch('/api/auth/logout', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });
        
        const result = await response.json();
        if (result.success) {
            window.location.href = '/';
        } else {
            alert('Logout failed. Please try again.');
        }
    } catch (error) {
        alert('Logout failed. Please try again.');
        console.error('Logout error:', error);
    }
}

// Load user usage data for dashboard
async function loadUserUsage() {
    try {
        const response = await fetch('/api/user/usage');
        if (response.ok) {
            const data = await response.json();
            updateUsageDisplay(data);
        }
    } catch (error) {
        console.error('Failed to load usage data:', error);
    }
}

// Update usage display on dashboard
function updateUsageDisplay(data) {
    const usageElement = document.getElementById('usage-stats');
    if (usageElement && data.usage) {
        const usage = data.usage;
        const plan = data.plan;
        
        usageElement.innerHTML = `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-white/80 dark:bg-gray-800/80 p-4 rounded-lg">
                    <h3 class="font-semibold text-gray-900 dark:text-white">Monthly Usage</h3>
                    <p class="text-2xl font-bold text-primary">${usage.monthly_usage || 0}</p>
                    <p class="text-sm text-gray-600 dark:text-gray-400">of ${plan.analyses_limit || 'unlimited'} analyses</p>
                </div>
                <div class="bg-white/80 dark:bg-gray-800/80 p-4 rounded-lg">
                    <h3 class="font-semibold text-gray-900 dark:text-white">Total Usage</h3>
                    <p class="text-2xl font-bold text-primary">${usage.total_analyses || 0}</p>
                    <p class="text-sm text-gray-600 dark:text-gray-400">all time</p>
                </div>
                <div class="bg-white/80 dark:bg-gray-800/80 p-4 rounded-lg">
                    <h3 class="font-semibold text-gray-900 dark:text-white">Current Plan</h3>
                    <p class="text-lg font-bold text-primary">${plan.name || 'Starter'}</p>
                    <p class="text-sm text-gray-600 dark:text-gray-400">$${plan.price || 19}/month</p>
                </div>
            </div>
        `;
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Load usage data if on dashboard
    if (window.location.pathname === '/dashboard') {
        loadUserUsage();
    }
    
    // Add event listeners for forms
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }
    
    const registerForm = document.getElementById('registerForm');
    if (registerForm) {
        registerForm.addEventListener('submit', handleRegister);
    }
    
    const paymentForm = document.getElementById('paymentForm');
    if (paymentForm) {
        paymentForm.addEventListener('submit', handlePayment);
    }
    
    // Add logout button listener
    const logoutButton = document.getElementById('logoutButton');
    if (logoutButton) {
        logoutButton.addEventListener('click', handleLogout);
    }
});
