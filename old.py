
def old_get_model_predictions(svm_parameters: SVM_Parameters, image_folder):
    classifier = load_svm(svm_parameters, "../saved_models")

    test_image_names = ['red.webp']

    SCALE_FACTOR = 1.25
    CONFIDENCE_THRESHOLD = 1
    results = {'test_images': [], 'detections': []}

    image_id = 0
    category_id = 1

    for image_name in tqdm(test_image_names):

        image_id += 1
        results['test_images'].append({
            "file_name": image_name,
            "image_id": image_id
        })

        image = cv2.imread(os.path.join(image_folder, image_name))

        original_image_height = image.shape[0]
        original_image_width = image.shape[1]

        # Resizing the image for speed purposes
        # Is that necessary?
        image = cv2.resize(image, (400, 256))
        resize_h_ratio = original_image_height / 256
        resize_w_ratio = original_image_width / 400

        image = get_grayscale_image(image)
        rects = []
        confidence = []

        scale = 0

        for scaled_image in pyramid_gaussian(image,downscale=SCALE_FACTOR):
            if(
                scaled_image.shape[0] < svm_parameters.window_size[0] and
                scaled_image.shape[1] < svm_parameters.window_size[1]
            ): break

            windows = sliding_window(scaled_image, svm_parameters.window_size, svm_parameters.step_size)
            for (x,y,window) in windows:
                # somethign def doesnt work here
                if window.shape[0] == svm_parameters.window_size[0] and window.shape[1] == svm_parameters.window_size[1]:
                    feature_vector = hog(
                        window,
                        svm_parameters.hog_parameters
                    )
                    feature_vector = feature_vector.reshape(1, -1) # or feature_vector = [feature_vector]
                    prediction = classifier.predict(feature_vector)
                    if prediction[0] == 1:
                        confidence_score = classifier.decision_function(feature_vector)

                        print(f"Confidence: {confidence_score[0]}")
                        if confidence_score > CONFIDENCE_THRESHOLD:
                            left_pos = int(x * (SCALE_FACTOR ** scale) * resize_w_ratio)
                            top_pos = int(y * (SCALE_FACTOR ** scale) * resize_h_ratio)

                            rects.append([
                                left_pos,
                                top_pos,
                                left_pos + original_image_width,
                                top_pos + original_image_height
                            ])
                            confidence.append([confidence_score])

            scale += 1

        # rects,scores = NMS(rects,confidence)
        # for rect,score in zip(rects,scores):
        #     x1,y1,x2,y2 = rect.tolist()
        #     results['detections'].append({"image_id":image_id,"category_id":category_id,"bbox":[x1,y1,x2-x1,y2-y1],"score":score.item()})

    return results


