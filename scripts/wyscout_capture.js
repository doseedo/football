/**
 * Wyscout Wide Angle Video URL Capture
 *
 * USAGE:
 * 1. Open Wyscout in your browser and login
 * 2. Open DevTools (F12) → Console tab
 * 3. Paste this entire script and press Enter
 * 4. Navigate to matches, click Export → Download Video → Start Download
 * 5. URLs are captured automatically and logged to console
 * 6. Type `downloadURLs()` to get all captured URLs
 * 7. Type `downloadAll()` to download all videos (browser will prompt)
 */

(function() {
    'use strict';

    // Store captured URLs
    window.wyscoutURLs = window.wyscoutURLs || [];

    // Intercept fetch requests
    const originalFetch = window.fetch;
    window.fetch = async function(...args) {
        const url = args[0]?.url || args[0];
        if (typeof url === 'string' && url.includes('cdn') && url.includes('.mp4')) {
            captureURL(url);
        }
        return originalFetch.apply(this, args);
    };

    // Intercept XHR requests
    const originalXHROpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url, ...rest) {
        if (url && url.includes('cdn') && url.includes('.mp4')) {
            captureURL(url);
        }
        return originalXHROpen.apply(this, [method, url, ...rest]);
    };

    // Watch for download link clicks
    document.addEventListener('click', function(e) {
        const link = e.target.closest('a[href*="cdn"][href*=".mp4"]');
        if (link) {
            captureURL(link.href);
        }
    }, true);

    // Monitor network requests via PerformanceObserver
    if (window.PerformanceObserver) {
        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.name.includes('cdn') && entry.name.includes('.mp4')) {
                    captureURL(entry.name);
                }
            }
        });
        observer.observe({ entryTypes: ['resource'] });
    }

    function captureURL(url) {
        if (!window.wyscoutURLs.includes(url)) {
            window.wyscoutURLs.push(url);
            const videoId = url.split('/').pop().split('?')[0];
            console.log(`%c✓ Captured #${window.wyscoutURLs.length}: ${videoId}`, 'color: #00ff00; font-weight: bold');
        }
    }

    // Helper functions
    window.downloadURLs = function() {
        console.log('\n=== CAPTURED URLS ===\n');
        window.wyscoutURLs.forEach((url, i) => {
            console.log(`${i + 1}. ${url}\n`);
        });
        console.log(`\nTotal: ${window.wyscoutURLs.length} URLs`);

        // Copy to clipboard
        const text = window.wyscoutURLs.join('\n');
        navigator.clipboard.writeText(text).then(() => {
            console.log('%c✓ URLs copied to clipboard!', 'color: #00ff00');
        });

        return window.wyscoutURLs;
    };

    window.downloadAll = function() {
        console.log(`Downloading ${window.wyscoutURLs.length} videos...`);
        window.wyscoutURLs.forEach((url, i) => {
            setTimeout(() => {
                const a = document.createElement('a');
                a.href = url;
                a.download = '';
                a.click();
            }, i * 2000); // 2 second delay between downloads
        });
    };

    window.clearURLs = function() {
        window.wyscoutURLs = [];
        console.log('%c✓ URLs cleared', 'color: #ffff00');
    };

    window.exportURLs = function() {
        const blob = new Blob([window.wyscoutURLs.join('\n')], {type: 'text/plain'});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'wyscout_urls.txt';
        a.click();
        console.log('%c✓ Exported to wyscout_urls.txt', 'color: #00ff00');
    };

    // Status
    console.log(`
%c╔════════════════════════════════════════════════════════════╗
║         WYSCOUT URL CAPTURE ACTIVE                         ║
╠════════════════════════════════════════════════════════════╣
║  Navigate to matches and click "Start Download"            ║
║  URLs will be captured automatically                       ║
║                                                            ║
║  Commands:                                                 ║
║    downloadURLs()  - Show & copy all captured URLs         ║
║    downloadAll()   - Download all videos                   ║
║    exportURLs()    - Save URLs to text file                ║
║    clearURLs()     - Clear captured URLs                   ║
║                                                            ║
║  Captured so far: ${window.wyscoutURLs.length.toString().padEnd(36)}║
╚════════════════════════════════════════════════════════════╝
`, 'color: #00ffff; font-weight: bold');

})();
