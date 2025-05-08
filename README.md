# RealTime-Sign-Language-Detection
✨ RealTime Sign Language Detection<br>
A real-time hand gesture and sign language recognition system using OpenCV, TensorFlow, and CVZone.<br>

📂 Project Structure

*dataCollection.py: Collects hand gesture images for training.<br>

*myClassifier.py: Custom classifier to load the trained model and predict gestures.<br>

*test.py: Runs real-time detection using webcam feed.<br>

*keras_model.h5: Trained model file.<br>

*labels.txt: Labels associated with gestures.<br>

🛠️ Installation<br>
1.Clone the repository:
```bash
git clone https://github.com/your-username/RealTime-SignLanguage-Detection.git
cd RealTime-SignLanguage-Detection
```
2.Create a virtual environment (recommended):
```bash
python -m venv myenv
myenv\Scripts\activate  # Windows
```
3.Install dependencies:
```bash
pip install -r requirements.txt
```
🚀 How to Run <br>
*To collect data:
```bash
python dataCollection.py
```
*To test real-time sign detection:
```bash
python test.py
```
🚀 Features:<br>
👍Real-time hand detection<br>

👌Custom sign language gesture classification<br>

🌪️FPS (Frames Per Second) counter for performance tracking<br>

😉Easy to extend with new signs<br>

📜 License<br>
This project is licensed under the MIT License — free to use and modify.

🌟 Credits<br>
Developed with ❤️ by Mahaboob Vali Shaik<br>

Special thanks to OpenCV, TensorFlow, and CVZone community.




