
document.addEventListener('DOMContentLoaded', function () {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    let questionIndex = -1; // Initialize question index to -1 (before the first question)

    const questions = [
        'What is your gender?',
        'What is your age?',
        'What is your BMI?',
        'What is your cholesterol level?',
        'What is your HDL-cholesterol level?',
        'What is your Hemoglobin A1c level?',
        'What are your creatinine levels?'
    ];

    // Initialize an array to store the answers
    const answers = [];

    let diagnose

    sendButton.addEventListener('click', function () {
        sendMessage();
    });


    userInput.addEventListener('keyup', function (event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    function displayUserMessage(message) {
        const userMessageElement = document.createElement('div');
        userMessageElement.className = 'user-message';
        userMessageElement.innerText = message;
        chatBox.appendChild(userMessageElement);
    }

    function displayBotMessage(message) {
        const botMessageElement = document.createElement('div');
        botMessageElement.className = 'bot-message';
        botMessageElement.innerText = message;
        chatBox.appendChild(botMessageElement);

        if (message === 'Hello! How can I assist you today?') {
            // Add buttons after the greeting message
            const buttons = document.createElement('div');
            buttons.className = 'button-container';
            buttons.innerHTML = `
                <button class="button" id="sick-button">I'm Sick</button>
                <button class="button" id="feel-great-button">I Feel Great</button>
            `;
            chatBox.appendChild(buttons);

            // Add event listeners for the buttons
            document.getElementById('sick-button').addEventListener('click', function () {
                displayBotMessage("I'm sorry to hear that. Please answer the following questions.");
                questionIndex = 0; // Start asking questions from index 0
                askQuestions();
            });

            document.getElementById('feel-great-button').addEventListener('click', function () {
                displayBotMessage("That's great to hear! If you have any questions in the future, feel free to ask.");
            });
        }
        }


        function sendMessage() {
            const userMessage = userInput.value.trim();
            if (userMessage !== '') {
                displayUserMessage(userMessage);
                userInput.value = '';
    
                setTimeout(function () {
                    const botResponse = getBotResponse(userMessage);
                    displayBotMessage(botResponse);
    
                    if (questionIndex >= 0 && questionIndex < questions.length) {
                        // Increment the question index
                        answers.push(userMessage);
                        questionIndex++;
                    }
                }, 800);
            }
        }
    
         function getBotResponse(userMessage) {
            if (userMessage.toLowerCase() === 'hi' || userMessage.toLowerCase() === 'hello') {
                return 'Hello! How can I assist you today?';
            } else if (questionIndex >= 0 && questionIndex <questions.length) {
                return questions[questionIndex]; 
            } else if (questionIndex === questions.length) {
                // Check if all questions have been answered
                //const thanksMessage = 'Thanks for answering!';
                getPrediction()
                return "Processing your test results...";
                
            } else {
                return "I'm sorry, I don't understand that.";
            }
    
    
        }
        

        async function getPrediction() {
            try {
                // Predict the diagnosis and wait for the result
                const prediction = await predict(answers);
        
                // Handle the prediction here
                diagnose = prediction;
                console.log(`Received prediction: ${prediction}`);
        
                // Update the chat with the diagnosis
                displayBotMessage(`You tested ${diagnose}`);
            } catch (error) {
                // Handle errors here
                console.error(`Prediction failed: ${error}`);
                displayBotMessage("An error occurred while predicting.");
            }
        }
    /*function askQuestions() {
        if (questionIndex >= 0 && questionIndex < questions.length) {
            displayBotMessage(questions[questionIndex]);
            questionIndex++
        }
    }*/
    function askQuestions() {
        if (questionIndex >= 0 && questionIndex < questions.length) {
            const currentQuestion = questions[questionIndex];
            displayBotMessage(currentQuestion);    
            questionIndex++;
        }
    }



  function predict(answers) {

    /*const inputData = {
        geslacht : answers[0],
        leeftijd :  answers[1], 
        creatinine : answers[5],
        hemoglobine_a1c :answers[4] ,
        cholesterol : answers[3],
        bmi : answers[2],
  };*/
  


    const inputData = {
        geslacht: 1,                // Gender (assuming 1 corresponds to Male)
        leeftijd: 26,   
        ureum: 3.5,                // Ureum (normal range)
        creatinine: 0.8,           // Creatinine (normal range)
        hemoglobine_a1c: 14,      // Hemoglobin A1c (normal range)
        cholesterol: 8,    
        triglyceriden: 1.0,     
        hdl_cholesterol: 1.2, 
        ldl_cholesterol: 2.3,
        vldl_cholesterol: 0.5 ,
        bmi: 30                  // BMI (normal range)

    };

  
   
    const inputArray = [Object.values(inputData)];

    return new Promise((resolve, reject) => {
    // Define the URL of your Flask API endpoint
    const apiUrl = 'http://127.0.0.1:5000/p'; // Adjust the URL as needed

    // Make a POST request to the API
    console.log('Requesting prediction...');

    fetch(apiUrl, {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputArray), // Send the 2D array as JSON data
    })
        .then((response) => {
        if (!response.ok) {
            throw new Error('Request failed');
        }
        return response.json();
        })
        .then((data) => {
            const diagnose = data.diagnose;
            console.log(`Prediction: ${diagnose}`);
            resolve(diagnose); // Resolve the promise with the prediction
        })
        .catch((error) => {
        console.error('Error:', error);
        reject(error); // Reject the promise with the error
        });
    })
}})
    
