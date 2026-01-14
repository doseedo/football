/**
 * Wyscout/Hudl Wide Angle Video URL Capture
 * Works on: wyscout.hudl.com
 *
 * Paste in DevTools Console (F12)
 */
(()=>{
    window.W=window.W||[];

    // Patterns to match video URLs (Hudl + Wyscout CDNs)
    const isVideoURL = (url) => {
        if (!url || typeof url !== 'string') return false;
        return (
            url.includes('.mp4') ||
            url.includes('.m3u8') ||
            url.includes('/video') ||
            url.includes('cdn') ||
            url.includes('hudl') && url.includes('video') ||
            url.includes('cloudfront') ||
            url.includes('akamai') ||
            url.includes('download') && (url.includes('wyscout') || url.includes('hudl'))
        );
    };

    // Intercept fetch
    const origFetch = window.fetch;
    window.fetch = async function(...args) {
        const url = args[0]?.url || args[0];
        if (isVideoURL(url)) capture(url, 'fetch');
        return origFetch.apply(this, args);
    };

    // Intercept XHR
    const origXHR = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url, ...rest) {
        if (isVideoURL(url)) capture(url, 'xhr');
        return origXHR.apply(this, [method, url, ...rest]);
    };

    // Watch clicks
    document.addEventListener('click', e => {
        const link = e.target.closest('a[href]');
        if (link && isVideoURL(link.href)) capture(link.href, 'click');
    }, true);

    // Watch all network requests
    if (window.PerformanceObserver) {
        new PerformanceObserver(list => {
            list.getEntries().forEach(e => {
                if (isVideoURL(e.name)) capture(e.name, 'network');
            });
        }).observe({entryTypes: ['resource']});
    }

    function capture(url, source) {
        // Skip duplicates and non-downloadable URLs
        if (W.some(u => u.url === url)) return;
        if (url.includes('thumbnail') || url.includes('poster')) return;

        W.push({url, source, time: new Date().toISOString()});
        const short = url.length > 80 ? url.substring(0, 80) + '...' : url;
        console.log(`%c✓ [${source}] Captured #${W.length}: ${short}`, 'color:#0f0; font-weight:bold');
    }

    // Helper functions
    window.urls = () => {
        console.log('\n=== CAPTURED URLS ===');
        W.forEach((item, i) => console.log(`${i+1}. [${item.source}] ${item.url}`));
        const urlList = W.map(u => u.url).join('\n');
        navigator.clipboard.writeText(urlList);
        console.log('%c✓ Copied to clipboard', 'color:#0f0');
        return W;
    };

    window.save = () => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([W.map(u=>u.url).join('\n')]));
        a.download = 'wyscout_urls.txt';
        a.click();
    };

    window.clear = () => { W.length = 0; console.log('Cleared'); };

    // Log all network activity for debugging
    window.debug = () => {
        const origFetch2 = window.fetch;
        window.fetch = async function(...args) {
            console.log('%c[FETCH]', 'color:yellow', args[0]?.url || args[0]);
            return origFetch2.apply(this, args);
        };
        console.log('Debug mode ON - all fetch requests will be logged');
    };

    console.log(`%c
╔═══════════════════════════════════════════════════════════╗
║  WYSCOUT/HUDL URL CAPTURE ACTIVE                          ║
╠═══════════════════════════════════════════════════════════╣
║  Commands:                                                ║
║    urls()  - Show all & copy to clipboard                 ║
║    save()  - Download as urls.txt                         ║
║    clear() - Clear captured URLs                          ║
║    debug() - Log ALL network requests                     ║
║                                                           ║
║  Captured: ${W.length} URLs                                        ║
╚═══════════════════════════════════════════════════════════╝
`, 'color:#0ff');
})();
