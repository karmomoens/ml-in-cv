from src.machinelearningsuite.machinelearningsuite import MachineLearningSuite


if __name__ == "__main__":
    # suite = MachineLearningSuite("../videos/trump.mp4", "../data/shape_predictor_68_face_landmarks.dat")
    suite = MachineLearningSuite("webcam", "./data/shape_predictor_68_face_landmarks.dat")
    suite.initialize()

    print(suite.configuration.to_dict())

    while True:
        print("MENU")
        print("=====")
        print("1. Create classes")
        print("2. Select the parts of the face to train on")
        print("3. Train")
        print("4. Predict")
        print("5. Reset configuration")
        print("6. Exit")
        choice = input("What do you want to do?")

        if choice == "1":
            suite.create_classes()
        elif choice == "2":
            suite.select_parts()
        elif choice == "3":
            suite.train()
        elif choice == "4":
            suite.predict()
        elif choice == "5":
            suite.configuration.reset()
        elif choice == "6":
            suite.quit()
        else:
            print("Wrong choice")
            continue
