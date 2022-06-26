# Interactive Digit AI

This project is composed by an interactive website built with Angular, and an AI build with Python. The AI is capable of
recognizing digits written by users in a canvas and print the results. Pyodide is used to run Python code in the
browser.
In this project, I decided not to use frameworks like Tensorflow, but to implement it from scratch in order to learn the
math behind Deep Learning.
The training process happens entirely on the browser, so everyone can contribute by simply clicking a button. The AI 
model is stored in public Firebase storage (just be careful of overfitting!).

## Features
* AI library built with numpy that includes:
  * Convolution Layer
  * MaxPool Layer
  * Dense Layer
  * Dropout Layer
  * ReLu Layer
  * etc.
* Deep learning using Gradient Descent, Cross Entropy Loss and ADAM Optimizer
* Python-Typescript integration using Pyodide
* UI built in Angular
  * Home: small introduction
  * Train: where users can train the AI model for a specified number of epochs
  * Digit recognition: where users can draw their digits on a canvas and the AI tries to guess it
* Firebase Storage integration
* Styling using SCSS and Angular Material
* Hosted page in GitHub Pages
* Auto deploy using GitHub Actions