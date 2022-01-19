import joblib
import cv2

model = joblib.load("models/model.pkl")


im_np = cv2.imread( "data/1.jpg", cv2.IMREAD_COLOR ) 
    
#  BGR 2 RGB
im_np = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)

# Resize
im_np = cv2.resize(im_np, (64,64))

# metto tutto su una riga
X = im_np.reshape(1,-1)


Y_hat = model.predict(X)

print(Y_hat)




