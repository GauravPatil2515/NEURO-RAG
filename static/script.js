// NeuroRAG Dashboard JavaScript

// Show/Hide sections
function showSection(sectionName) {
    // Hide all sections
    document.getElementById('dashboardSection').style.display = 'none';
    document.getElementById('databaseSection').style.display = 'none';
    
    // Remove active class from all links
    document.querySelectorAll('.link-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Show selected section
    if (sectionName === 'dashboard') {
        document.getElementById('dashboardSection').style.display = 'block';
        document.querySelectorAll('.link-item')[0].classList.add('active');
    } else if (sectionName === 'database') {
        document.getElementById('databaseSection').style.display = 'block';
        document.querySelectorAll('.link-item')[1].classList.add('active');
    }
}

// Load database content
async function loadDatabase() {
    const contentDiv = document.getElementById('databaseContent');
    contentDiv.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 2rem;">Loading database...</p>';
    
    try {
        const response = await fetch('/api/database');
        const data = await response.json();
        
        if (data.success) {
            contentDiv.innerHTML = data.content;
        } else {
            contentDiv.innerHTML = `<p style="color: var(--error);">Error: ${data.error}</p>`;
        }
    } catch (error) {
        contentDiv.innerHTML = `<p style="color: var(--error);">Failed to load database: ${error.message}</p>`;
    }
}

// Set query from suggestion chip
function setQuery(text) {
    document.getElementById('searchInput').value = text;
    document.getElementById('searchInput').focus();
}

// Handle Enter key press
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        performSearch();
    }
}

// Perform search
async function performSearch() {
    const input = document.getElementById('searchInput');
    const query = input.value.trim();
    
    if (!query) {
        alert('Please enter a question');
        return;
    }
    
    // Show loading state
    const btn = document.getElementById('searchBtn');
    const btnText = document.getElementById('searchBtnText');
    const loader = document.getElementById('searchLoader');
    
    btn.disabled = true;
    btnText.style.display = 'none';
    loader.style.display = 'inline-block';
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.query, data.result);
        } else {
            displayError(data.error || 'An error occurred');
        }
        
    } catch (error) {
        displayError('Failed to connect to server: ' + error.message);
    } finally {
        // Reset button state
        btn.disabled = false;
        btnText.style.display = 'inline';
        loader.style.display = 'none';
    }
}

// Display results
function displayResults(query, result) {
    const resultsCard = document.getElementById('resultsCard');
    const resultsContent = document.getElementById('resultsContent');
    
    resultsContent.innerHTML = `
        <div style="margin-bottom: 1rem;">
            <strong style="color: var(--primary-green);">Query:</strong> 
            <span style="color: var(--text-primary);">${escapeHtml(query)}</span>
        </div>
        <div style="border-top: 2px solid var(--border-color); padding-top: 1rem;">
            ${formatResult(result)}
        </div>
    `;
    
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Format result text
function formatResult(text) {
    // Escape HTML
    let formatted = escapeHtml(text);
    
    // Format headers (lines starting with 📚, 💡, etc.)
    formatted = formatted.replace(/^(📚|💡|❌|⚠️)(.*?)$/gm, '<strong style="color: var(--primary-green);">$1$2</strong>');
    
    // Format separators
    formatted = formatted.replace(/---/g, '<hr style="border: 1px solid var(--border-color); margin: 1rem 0;">');
    
    // Format bold text
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    return formatted;
}

// Clear results
function clearResults() {
    const resultsCard = document.getElementById('resultsCard');
    resultsCard.style.display = 'none';
    document.getElementById('searchInput').value = '';
    document.getElementById('searchInput').focus();
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Load system stats on page load
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.vectorstore_loaded) {
            document.getElementById('vectorStatus').textContent = 'Loaded';
            document.getElementById('statusBadge').innerHTML = `
                <span class="dot"></span>
                <span>System Online</span>
            `;
        } else {
            document.getElementById('vectorStatus').textContent = 'Not Loaded';
            document.getElementById('statusBadge').innerHTML = `
                <span class="dot" style="background: var(--warning);"></span>
                <span>Loading...</span>
            `;
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
        document.getElementById('statusBadge').innerHTML = `
            <span class="dot" style="background: var(--error);"></span>
            <span>Offline</span>
        `;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    document.getElementById('searchInput').focus();
});
