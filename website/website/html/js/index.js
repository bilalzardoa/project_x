// Get references to the button and modal elements
const previewButton = document.querySelector('.preview-button');
const modal = document.querySelector('.modal');
const closeButton = document.querySelector('.close');

// Function to open the modal
function openModal() {
    modal.style.display = 'block';
}

// Function to close the modal
function closeModal() {
    modal.style.display = 'none';
}

// Event listener to open the modal when the button is clicked
previewButton.addEventListener('click', openModal);

// Event listener to close the modal when the close button is clicked
closeButton.addEventListener('click', closeModal);

// Event listener to close the modal when clicking outside the modal
window.addEventListener('click', (event) => {
    if (event.target === modal) {
        closeModal();
    }
});


const slider = document.querySelector('.industries-slider');
const prevButton = document.getElementById('prev-button');
const nextButton = document.getElementById('next-button');

let scrollAmount = 0;
const scrollStep = 200;

prevButton.addEventListener('click', () => {
    scrollAmount += scrollStep;
    if (scrollAmount > 0) {
        scrollAmount = 0;
    }
    slider.style.transform = `translateX(${scrollAmount}px)`;
});

nextButton.addEventListener('click', () => {
    scrollAmount -= scrollStep;
    const maxScroll = -(slider.scrollWidth - slider.clientWidth);
    if (scrollAmount < maxScroll) {
        scrollAmount = maxScroll;
    }
    slider.style.transform = `translateX(${scrollAmount}px)`;
});
