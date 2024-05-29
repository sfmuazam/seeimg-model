import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from io import BytesIO
from typing import Annotated, Union
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from starlette.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
import tensorflow as tf
import pickle
import heapq

from models import *

app = FastAPI(title="Img2Word",
              description="Dari gambar ke kalimat",)

def custom_openapi():
    if not app.openapi_schema:
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        )
        # Remove 422 responses from the schema
        for path in openapi_schema['paths']:
            for method in openapi_schema['paths'][path]:
                if '422' in openapi_schema['paths'][path][method]['responses']:
                    del openapi_schema['paths'][path][method]['responses']['422']
        app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3333",
    "http://localhost:3300",
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:3333",
    "http://127.0.0.1:3300",
    "http://127.0.0.1:5500","https://sfmuazam.github.io/seeimg"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Load the saved tokenizer
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))

# Define model parameters
top_k = 1566
num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
row_size = 8
col_size = 8
target_vocab_size = top_k + 1
dropout_rate = 0.1

# Initialize the Transformer model
transformer = Transformer(num_layer,
                          d_model,
                          num_heads,
                          dff,
                          row_size,
                          col_size,
                          target_vocab_size,
                          max_pos_encoding=target_vocab_size,
                          rate=dropout_rate)

# Dummy call to create the model variables
dummy_input = tf.random.uniform((1, row_size * col_size, 2048))
dummy_target = tf.random.uniform((1, 1), maxval=target_vocab_size, dtype=tf.int32)
_ = transformer(dummy_input, dummy_target, training=False)

# Load pre-trained weights
transformer.load_weights('img2cap_model.h5')

# Valid image formats
VALID_IMAGE_FORMATS = {"image/jpeg", "image/png", "image/webp", "jpg", "jpeg", "png", "webp"}

# Load image and preprocess
def load_image_from_file(file: UploadFile):
    try:
        img = Image.open(BytesIO(file.file.read()))
        if img.format.lower() not in VALID_IMAGE_FORMATS:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Format gambar tidak valid. Format yang diterima: jpg, jpeg, webp, png.")
        
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img = img.resize((224, 224))
        img = np.array(img)
        img = tf.keras.applications.resnet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img
    except UnidentifiedImageError:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="File gambar tidak valid.")

# Create masks for the decoder
def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask

# Evaluate the image and generate the caption
def evaluate(image_tensor):
    img_tensor_val = image_features_extract_model(image_tensor)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']

    # Decoder input is the start token.
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0)  # tokens
    result = []  # word list

    for i in range(100):
        dec_mask = create_masks_decoder(output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(img_tensor_val, output, False, dec_mask)

        # Select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # Return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer.word_index['<end>']:
            return result, tf.squeeze(output, axis=0), attention_weights

        # Concatenate the predicted_id to the output which is given to the decoder as its input.
        result.append(tokenizer.index_word[int(predicted_id)])
        output = tf.concat([output, predicted_id], axis=-1)

    return result, tf.squeeze(output, axis=0), attention_weights

def beam_search_decoder(predictions, beam_width):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in predictions:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -np.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select beam_width best
        sequences = ordered[:beam_width]
    return sequences

def evaluate_beam_search(image_tensor, beam_width=3):
    img_tensor_val = image_features_extract_model(image_tensor)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']

    # Decoder input is the start token.
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0)  # tokens

    result = []  # word list

    for i in range(100):
        dec_mask = create_masks_decoder(output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(img_tensor_val, output, False, dec_mask)
        predictions = tf.nn.softmax(predictions, axis=-1).numpy()

        # Beam search
        sequences = beam_search_decoder(predictions[:, -1, :], beam_width)

        predicted_id = sequences[0][0][-1]  # select the best sequence

        if predicted_id == end_token:
            return result, tf.squeeze(output, axis=0), attention_weights

        result.append(tokenizer.index_word[predicted_id])
        output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)

    return result, tf.squeeze(output, axis=0), attention_weights

async def correct_caption(caption: str) -> str:
    url = "https://api.nyx.my.id/ai/gpt4"
    query_params = {"text": f"Perbaiki teks berikut dan tambahkan tanda baca yang sesuai: \"{caption}\""}
    response = requests.get(url, params=query_params)
    if response.status_code == 200:
        return response.json().get("result", caption)
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Gagal memperbaiki caption.")
    
# Response Models
class ApiResponse(BaseModel):
    success: bool
    message: str
    caption: Union[str, None]

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": "Data yang diberikan tidak valid.",
            "caption": None
        }
    )

@app.post("/generate/", response_model=ApiResponse, responses={
    200: {
        "description": "Proses Berhasil",
        "content": {
            "application/json": {
                "example": {
                    "success": True,
                    "message": "Deskripsi gambar berhasil dihasilkan.",
                    "caption": "Ini adalah contoh deskripsi gambar."
                }
            }
        }
    },
    415: {
        "description": "Validasi Error",
        "content": {
            "application/json": {
                "example": {
                    "success": False,
                    "message": "Format gambar tidak valid. Format yang diterima: jpg, jpeg, webp, png.",
                    "caption": None
                }
            }
        }
    },
    500: {
        "description": "Kesalahan Server",
        "content": {
            "application/json": {
                "example": {
                    "success": False,
                    "message": "Terjadi kesalahan pada server.",
                    "caption": None
                }
            }
        }
    }
})
async def generate(file: Annotated[UploadFile, File(description="File gambar yang diunggah")]):
    if file.content_type.split("/")[-1] not in VALID_IMAGE_FORMATS:
        return JSONResponse(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, content={
            "success": False,
            "message": "Format gambar tidak valid. Format yang diterima: jpg, jpeg, webp, png.",
            "caption": None
        })
    
    try:
        image_tensor = load_image_from_file(file)
        caption, _, _ = evaluate_beam_search(image_tensor)
        caption = ' '.join([word for word in caption if word != "<unk>"])
        corrected_caption = await correct_caption(caption)
        return JSONResponse(content={
            "success": True,
            "message": "Deskripsi gambar berhasil dihasilkan.",
            "caption": corrected_caption
        })
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={
            "success": False,
            "message": e.detail,
            "caption": None
        })
    except Exception as e:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={
            "success": False,
            "message": "Terjadi kesalahan pada server.",
            "caption": None
        })