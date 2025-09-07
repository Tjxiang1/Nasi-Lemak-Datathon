// Add fade-in animation to elements as they come into view
document.addEventListener('DOMContentLoaded', function() {
    const elements = document.querySelectorAll('.fade-in');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    });

    elements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });

    // Form submission feedback
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const button = this.querySelector('.btn');
            const originalText = button.textContent;
            button.textContent = '⏳ Analyzing...';
            button.disabled = true;
            
            // Re-enable after 3 seconds (in case of errors)
            setTimeout(() => {
                button.textContent = originalText;
                button.disabled = false;
            }, 3000);
        });
    });

    // Modal functionality for charts
    const modal = document.getElementById('chartModal');
    const modalImage = document.getElementById('modalImage');
    const modalTitle = document.getElementById('modalTitle');
    const closeBtn = document.getElementsByClassName('close')[0];

    // Add click event to all chart images
    const chartImages = document.querySelectorAll('.chart-container img');
    chartImages.forEach(img => {
        img.addEventListener('click', function() {
            modal.style.display = 'block';
            modalImage.src = this.src;
            
            // Get the chart title from the parent card
            const cardTitle = this.closest('.card').querySelector('h3');
            if (cardTitle) {
                modalTitle.textContent = cardTitle.textContent;
            } else {
                modalTitle.textContent = 'Chart View';
            }
            
            // Prevent body scroll when modal is open
            document.body.style.overflow = 'hidden';
        });
    });

    // Close modal when clicking the X
    closeBtn.addEventListener('click', function() {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    });

    // Close modal when clicking outside the modal content
    modal.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    });

    // Close modal with Escape key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && modal.style.display === 'block') {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    });
});