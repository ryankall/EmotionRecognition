import cv2
import json
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold

from scipy.stats import sem
from sklearn import metrics
import numpy as np
from scipy.ndimage import zoom
from sklearn import datasets


# Get label data form xml file
def load_table():
    table = json.load(open("label_data.xml"))
    return table


# Trained classifier's performance evaluation
def cross_validation(classifier, X, y, K):
    # create a k-fold cross validation iterator and shuffles the data
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(classifier, X, y, cv=cv)
    print "Scores: ", (scores)
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


# Confusion Matrix and Results
def train_and_evaluate(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    print ("Accuracy on training set:")
    print (classifier.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (classifier.score(X_test, y_test))
    y_pred = classifier.predict(X_test)
    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))


# convert keys from string to int
def convert_table(xml_table):
    new_table = {}
    for i in xml_table:
        new_table[int(i)] = xml_table[i]
    return new_table


# print subject in training data
def print_subject_count(arr):
    print "Number of happy subjects: ",
    print np.count_nonzero(arr)
    print "Number of unhappy subjects: ",
    print arr.size - np.count_nonzero(arr)
    print "Total subject: ",
    print arr.size


# zoom into the face
def extract_face_(gray, x, y, w, h, offset_coefficients):
    horizontal_offset = int(offset_coefficients[0] * w)
    vertical_offset = int(offset_coefficients[1] * h)
    extracted_face = gray[y + vertical_offset:y + h,
                     x + horizontal_offset:x - horizontal_offset + w]
    # zoom in
    new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0],
                                               64. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(np.float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face


if __name__ == '__main__':

    # gets image database from sklearn
    faces = datasets.fetch_olivetti_faces()

    # get label data
    temp_table = load_table()
    table = convert_table(temp_table)

    # indexing the image data
    data = faces.data[[int(i) for i in table], :]

    # convert true and false to binary
    target = np.array([table[i] for i in table]).astype(np.int32)

    print_subject_count(target)

    # Train the classifier
    # split the the data for 20% for test and rest for training
    assert isinstance(data, object)
    p1_train, p1_test, p2_train, p2_test = train_test_split(data, target, test_size=0.20,
                                                            random_state=0)

    # initialize classifier
    supportVC = SVC(kernel='linear')

    # 10 fold cross validation
    cross_validation(supportVC, p1_train, p2_train, 10)

    # train the data
    train_and_evaluate(supportVC, p1_train, p1_test, p2_train, p2_test)

    # load face classifier form open cv2 to detect face
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # start video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.putText(frame, "Press q to QUIT", (frame.shape[1] - 200, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # get face in frame
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(50, 50),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:

            # put a green square around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # zoom into the face
            found_face = extract_face_(gray, x, y, w, h, (.3, 0.05))

            if supportVC.predict(found_face.reshape(1, -1)):
                cv2.putText(frame, "Happy", (int(frame.shape[1] / 2), frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            155, 5)
            else:
                cv2.putText(frame, "Not Happy", (int(frame.shape[1] / 2), frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 6)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
