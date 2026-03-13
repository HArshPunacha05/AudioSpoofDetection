import gradio as gr
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Model Architecture
# ----------------------------

class SpoofCNN(nn.Module):

    def __init__(self):

        super(SpoofCNN, self).__init__()

        self.conv1 = nn.Conv2d(1,16,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(51200,128)
        self.fc2 = nn.Linear(128,2)


    def forward(self,x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x


# ----------------------------
# Load trained model
# ----------------------------

model = SpoofCNN()

model.load_state_dict(
    torch.load("asvspoof_cnn_model.pth", map_location="cpu")
)

model.eval()


# ----------------------------
# Prediction Function
# ----------------------------

def predict_audio(file):

    if file is None:
        return "⚠️ Please upload an audio file.", None


    audio, sr = librosa.load(file, sr=16000)


    # Create Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel)


    # Spectrogram Visualization
    fig, ax = plt.subplots()

    img = ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower"
    )

    ax.set_title("Mel Spectrogram")

    ax.set_xlabel("Time")

    ax.set_ylabel("Mel Frequency")

    plt.colorbar(img)


    # Convert to tensor
    mel_tensor = torch.tensor(mel_db).float()

    max_len = 200


    if mel_tensor.shape[1] < max_len:

        pad = max_len - mel_tensor.shape[1]

        mel_tensor = F.pad(mel_tensor,(0,pad))

    else:

        mel_tensor = mel_tensor[:,:max_len]


    # Add batch + channel dimension
    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)


    with torch.no_grad():

        output = model(mel_tensor)

        probs = torch.softmax(output,dim=1)

        pred = torch.argmax(probs,dim=1).item()

        confidence = probs[0][pred].item()


    if pred == 0:

        label = "🟢 BONAFIDE (Real Voice)"

        color = "green"

    else:

        label = "🔴 SPOOF (Fake Voice)"

        color = "red"


    html_result = f"""
    <div style="font-size:25px;font-weight:bold;color:{color}">
    {label}
    </div>
    <div style="font-size:18px">
    Confidence: {confidence*100:.2f}%
    </div>
    """


    return html_result, fig



# ----------------------------
# Gradio Interface
# ----------------------------

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🎙️ Voice Deepfake Detection System")

    gr.Markdown(
        "Upload audio to detect whether the voice is real or spoofed."
    )


    with gr.Row():

        audio_input = gr.Audio(
            sources = ["upload", "microphone"],
            type="filepath",
            label="Upload or Record Audio"
        )


    predict_btn = gr.Button("🔍 Analyze Voice")


    result = gr.HTML()

    spectrogram = gr.Plot()


    predict_btn.click(

        fn=predict_audio,

        inputs=audio_input,

        outputs=[result, spectrogram]

    )


demo.launch()
