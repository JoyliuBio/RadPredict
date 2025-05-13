function formatUTCtoLocal(utcTimeString) {
    const date = new Date(utcTimeString);
    
    const chinaTime = new Date(date.getTime() + (8 * 60 * 60 * 1000));
    
    return chinaTime.toLocaleString('zh-CN', {
        year: 'numeric', 
        month: '2-digit', 
        day: '2-digit',
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit',
        hour12: false
    });
}

document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.utc-time').forEach(el => {
        const timestamp = el.getAttribute('data-timestamp');
        if (timestamp) {
            el.textContent = formatUTCtoLocal(timestamp);
        }
    });
}); 