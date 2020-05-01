import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1hTF9uPjao9HogRWlCdy4QoQFWpjHQB7K'
export_file_name = 'export.pkl'

classes = ['Apple___Apple_scab', 'Tomato___Late_blight', 'Tomato___Septoria_leaf_spot', 'Pepper_bell___Bacterial_spot', 'Grape___Esca_Black_Measles', 'Tomato___Bacterial_spot', 'Blueberry___healthy', 'Cherry_including_sour___healthy', 'Corn_maize___healthy', 'Raspberry___healthy', 'Apple___healthy', 'Grape___Leaf_blight_Isariopsis_Leaf_Spot', 'Grape___healthy', 'Tomato___Leaf_Mold', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Strawberry___Leaf_scorch', 'Tomato___healthy', 'Grape___Black_rot', 'Corn_maize___Cercospora_leaf_spot Gray_leaf_spot', 'Peach___healthy', 'Peach___Bacterial_spot', 'Tomato___Target_Spot', 'Squash___Powdery_mildew', 'Apple___Cedar_apple_rust', 'Potato___healthy', 'Orange___Haunglongbing_Citrus_greening', 'Tomato___Early_blight', 'Cherry_including_sour___Powdery_mildew', 'Soybean___healthy', 'Tomato___Spider_mites_Two-spotted_spider_mite', 'Potato___Early_blight', 'Potato___Late_blight', 'Pepper_bell___healthy', 'Strawberry___healthy', 'Corn_maize___Northern_Leaf_Blight', 'Corn_maize___Common_rust_', 'Apple_Frogeye_Spot']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
