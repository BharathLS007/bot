document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById('contactForm');
    form.addEventListener('submit', function (e) {
        e.preventDefault();

        const formData = {
            name: document.getElementById('name').value,
            email: document.getElementById('email').value,
            subject: document.getElementById('subject').value,
            department: document.getElementById('department').value,
            message: document.getElementById('message').value
        };

        fetch('/submit-form', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                document.getElementById('thankYouMsg').style.display = 'block';
                form.reset();
                setTimeout(() => {
                    document.getElementById('thankYouMsg').style.display = 'none';
                }, 5000);
            }
        })
        .catch(err => console.error('Form submission error:', err));
    });
});
