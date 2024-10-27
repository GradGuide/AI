let selectedAlgorithm = 'cosine-similarity';

function toggleDropdown() {
    document.getElementById("myDropdown").classList.toggle("show");
}

function selectAlgorithm(algorithm) {
    selectedAlgorithm = algorithm;
    document.getElementById("myDropdown").classList.remove("show");
    document.querySelector('.dropdown button').textContent = algorithm.toUpperCase();
}

function sendRequest() {
    const text1 = document.getElementById('text1').value;
    const text2 = document.getElementById('text2').value;

    const requestBody = {
        text1: text1,
        text2: text2,
        alg: selectedAlgorithm
    };

    fetch('http://127.0.0.1:5000/api/compare', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
    })
        .then(response => response.json())
        .then(data => {
            // Assuming the response contains a similarity score as a string
            const similarityScore = parseFloat(data.similarity) * 100; // Convert string to float and multiply by 100
            document.getElementById('percentage').textContent = `${similarityScore.toFixed(2)}%`; // Display percentage with two decimal places
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Close the dropdown if the user clicks outside of it
window.onclick = function(event) {
    if (!event.target.matches('.dropdown button')) {
        const dropdowns = document.getElementsByClassName("dropdown-content");
        for (let i = 0; i < dropdowns.length; i++) {
            const openDropdown = dropdowns[i];
            if (openDropdown.classList.contains('show')) {
                openDropdown.classList.remove('show');
            }
        }
    }
}
