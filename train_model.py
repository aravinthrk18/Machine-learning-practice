import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("student_data.csv")

# Features (inputs)
X = data[['Hours','Attendance','InternalMarks']]

# Target (output)
y = data['Result']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train,y_train)

# Save model
joblib.dump(model, "student_model.pkl")

# Print learned weights
print("\nLearned Importance of Features:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef}")

print("Intercept:", model.intercept_[0])

# Test accuracy
pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test,pred))

# ----------- Prediction -----------
hours = float(input("\nEnter study hours: "))
attendance = float(input("Enter attendance percentage: "))
marks = float(input("Enter internal marks: "))

new_student = [[hours, attendance, marks]]
result = model.predict(new_student)

if result[0]==1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
