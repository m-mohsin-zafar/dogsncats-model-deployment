from commons import get_tensor, get_model

model = get_model()

class_to_name = {
    '0': 'cat',
    '1': 'dog'
}


def get_prediction(image_bytes):
    try:
        tensor_img = get_tensor(image_bytes)
        outputs = model.forward(tensor_img)
    except Exception:
        return 0, 'error!'
    conf, prediction = outputs.max(1)
    pet_class = prediction.item()
    pet_name = class_to_name[str(pet_class)]
    return pet_class, pet_name
