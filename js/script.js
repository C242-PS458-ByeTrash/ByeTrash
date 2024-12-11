const signUpButton = document.getElementById('signUpButton');
const signInButton = document.getElementById('signInButton');
const signInForm = document.getElementById('signIn');
const signUpForm = document.getElementById('signup');

signUpButton.addEventListener('click', function() {
    signInForm.style.display = "none";
    signUpForm.style.display = "block";
});

signInButton.addEventListener('click', function() {
    signInForm.style.display = "block";
    signUpForm.style.display = "none";
});

const resultContainer = document.getElementById('result');
const uploadBox = document.getElementById('uploadBox');
uploadBox.innerHTML = "<p>Drop image or click to select</p><p>JPG, PNG, BMP, or WEBP</p>";

// Ensure 'result' is defined before using it
const result = {
  prediction: 'exampleLabel',
  confidence: 0.95
};

if (result) {
  resultContainer.innerHTML = `<h3>Prediction Result:</h3>
    <p><strong>Label:</strong> ${result.prediction}</p>
    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>`;
}
