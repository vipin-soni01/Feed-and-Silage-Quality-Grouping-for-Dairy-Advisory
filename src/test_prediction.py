from predict import predict_quality

sample = {
    "dm.s": 32.0,
    "ash.s": 3.6,
    "cp.s": 7.2,
    "ee.s": 2.3,
    "ndf.s": 40.0,
    "adf.s": 21.0,
    "starch.s": 30.0,
    "pH": 3.8,
    "ammonia.s": 5.0,
    "glucose.s": 0.5,
    "fructose.s": 0.3,
    "ethanol.s": 1.0,
    "lactic.ac.s": 5.0,
    "acetic.ac.s": 1.2,
    "propionic.ac.s": 0.1,
    "butyric.ac.s": 0.06,
    "dm.loss": 8.0
}

result = predict_quality(sample)

print(result)