document.getElementById('contactForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const formData = {
        name: document.getElementById('name').value,
        email: document.getElementById('email').value,
        subject: document.getElementById('subject').value,
        department: document.getElementById('department').value,
        message: document.getElementById('message').value
    };

    // Convert to JSON and download as a file
    const blob = new Blob([JSON.stringify(formData, null, 4)], { type: "application/json" });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = "form_submission.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Optionally show thank you message
    document.getElementById('thankYouMsg').style.display = 'block';
    document.getElementById('contactForm').reset();
});