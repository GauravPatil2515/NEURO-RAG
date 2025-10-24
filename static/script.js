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
    const searchInput = document.getElementById('dbSearch');
    
    contentDiv.innerHTML = '<p style="text-align: center; color: var(--accent-green); padding: 2rem;"><span class="loader"></span> Loading database...</p>';
    
    try {
        const response = await fetch('/api/database');
        const data = await response.json();
        
        if (data.success) {
            // Store full content for searching
            window.dbFullContent = data.content;
            
            // Show first 10000 characters as preview
            const preview = data.content.substring(0, 10000);
            const isPreview = data.content.length > 10000;
            
            contentDiv.innerHTML = `
                <div style="color: var(--text-primary); white-space: pre-wrap; font-family: monospace; font-size: 0.9rem; line-height: 1.6;">${escapeHtml(preview)}${isPreview ? '\n\n...(use search to find specific content)' : ''}</div>
            `;
            
            // Show stats
            const stats = document.createElement('div');
            stats.style.cssText = 'margin-top: 1rem; padding: 1rem; background: var(--bg-card); border-radius: 8px; font-size: 0.875rem; color: var(--text-secondary);';
            stats.innerHTML = `
                <strong style="color: var(--accent-green);">✅ Database Loaded</strong><br>
                <strong>Total size:</strong> ${data.total_length.toLocaleString()} characters | 
                <strong>Lines:</strong> ${data.lines.toLocaleString()} | 
                <strong>Showing:</strong> ${preview.length.toLocaleString()} characters (preview)
            `;
            contentDiv.insertBefore(stats, contentDiv.firstChild);
            
            // Enable search
            searchInput.disabled = false;
            searchInput.placeholder = 'Search for ICD codes, disorders, symptoms...';
        } else {
            contentDiv.innerHTML = `<p style="color: var(--error); padding: 2rem;">❌ Error: ${escapeHtml(data.error)}</p>`;
        }
    } catch (error) {
        contentDiv.innerHTML = `<p style="color: var(--error); padding: 2rem;">❌ Failed to load database: ${escapeHtml(error.message)}</p>`;
    }
}

// Search database content
function searchDatabase() {
    const searchInput = document.getElementById('dbSearch');
    const contentDiv = document.getElementById('databaseContent');
    const searchTerm = searchInput.value.trim().toLowerCase();
    
    if (!window.dbFullContent) {
        alert('⚠️ Please load the database first by clicking "Load Data"');
        return;
    }
    
    if (!searchTerm) {
        // Show preview if search is empty
        const preview = window.dbFullContent.substring(0, 10000);
        const isPreview = window.dbFullContent.length > 10000;
        
        contentDiv.innerHTML = `
            <div style="margin-bottom: 1rem; padding: 1rem; background: var(--bg-card); border-radius: 8px; font-size: 0.875rem; color: var(--text-secondary);">
                <strong style="color: var(--accent-green);">✅ Database Loaded</strong><br>
                <strong>Total size:</strong> ${window.dbFullContent.length.toLocaleString()} characters
            </div>
            <div style="color: var(--text-primary); white-space: pre-wrap; font-family: monospace; font-size: 0.9rem; line-height: 1.6;">${escapeHtml(preview)}${isPreview ? '\n\n...(use search to find specific content)' : ''}</div>
        `;
        return;
    }
    
    // Find matches
    const lines = window.dbFullContent.split('\n');
    const matches = [];
    
    lines.forEach((line, index) => {
        if (line.toLowerCase().includes(searchTerm)) {
            matches.push({
                lineNum: index + 1,
                content: line
            });
        }
    });
    
    if (matches.length > 0) {
        let html = `<div style="margin-bottom: 1rem; padding: 1rem; background: var(--accent-green); color: white; border-radius: 8px; font-weight: 600;">
            🔍 Found ${matches.length} match${matches.length > 1 ? 'es' : ''} for "${escapeHtml(searchTerm)}"
        </div>`;
        
        const displayLimit = Math.min(matches.length, 200);
        
        matches.slice(0, displayLimit).forEach(match => {
            const highlighted = match.content.replace(
                new RegExp(escapeRegex(searchTerm), 'gi'),
                m => `<mark style="background: #ffeb3b; padding: 2px 4px; border-radius: 3px; color: #000; font-weight: 600;">${m}</mark>`
            );
            html += `<div style="margin-bottom: 0.75rem; padding: 0.75rem; border-left: 3px solid var(--accent-green); background: var(--bg-card); border-radius: 4px;">
                <div style="font-size: 0.7rem; color: var(--text-secondary); margin-bottom: 0.25rem; font-family: monospace;">Line ${match.lineNum}</div>
                <div style="color: var(--text-primary); white-space: pre-wrap; font-family: monospace; font-size: 0.85rem; line-height: 1.5;">${highlighted}</div>
            </div>`;
        });
        
        if (matches.length > displayLimit) {
            html += `<p style="text-align: center; color: var(--text-secondary); padding: 1rem; background: var(--bg-card); border-radius: 8px;">
                📄 Showing first ${displayLimit} of ${matches.length} matches
            </p>`;
        }
        
        contentDiv.innerHTML = html;
    } else {
        contentDiv.innerHTML = `
            <div style="text-align: center; padding: 3rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                <p style="color: var(--text-secondary); font-size: 1.1rem;">No matches found for <strong>"${escapeHtml(searchTerm)}"</strong></p>
                <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">Try different keywords or check spelling</p>
            </div>
        `;
    }
}

// Helper function to escape regex special characters
function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
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
