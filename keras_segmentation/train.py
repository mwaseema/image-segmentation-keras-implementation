import json
import os

import six
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau

from . import custom_losses
from .custom_losses import smooth_l1_loss
from .data_utils.bounding_box_based_network_utils import bounding_box_based_network_loss_gpu
from .data_utils.bounding_box_iou_based_network_utils import bounding_box_iou_based_network_loss, \
    bounding_box_iou_based_network_metric
from .data_utils.data_loader import image_segmentation_generator, IoU_network_image_segmentation_generator, \
    verify_segmentation_dataset, two_stream_verify_segmentation_dataset, two_stream_image_segmentation_generator, \
    image_segmentation_generator_i3d, image_segmentation_generator_bounding_box_based_network, \
    image_segmentation_generator_bounding_box_iou_based_network, image_segmentation_generator_with_weighted_output, \
    image_segmentation_generator_i3d_inception, image_segmentation_temporal_generator_with_weighted_output
from .data_utils.iou_utils import iou_metric_wrapper
from .models import model_from_name


def find_latest_checkpoint(checkpoints_path):
    ep = 0
    r = None
    while True:
        if os.path.isfile(checkpoints_path + "." + str(ep)):
            r = checkpoints_path + "." + str(ep)
        else:
            return r

        ep += 1


def replace_previous_checkpoint_with_empty_file(checkpoint_path, epoch_number):
    if epoch_number > 0:
        previous_check_point_path = f"{checkpoint_path}.{epoch_number - 1}"
        if os.path.exists(previous_check_point_path):
            os.remove(previous_check_point_path)
            # make new empty file
            f = open(previous_check_point_path, mode='w')
            f.close()


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          optimizer_name='adam'
          ):
    if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
        # create the model from the name
        assert (not n_classes is None), "Please provide the n_classes"
        if (not input_height is None) and (not input_width is None):
            model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert not (val_images is None)
        assert not (val_annotations is None)

    if not optimizer_name is None:
        # model.compile(loss='categorical_crossentropy',
        # 	optimizer= optimizer_name ,
        # 	metrics=['accuracy'])

        # model.compile(loss=[custom_losses.categorical_focal_loss_with_iou(alpha=0.50, gamma=1.25, model=model)],
        #               optimizer=optimizer_name,
        #               metrics=['accuracy', iou_metric_wrapper(output_height, output_width, n_classes)])

        model.compile(loss=[custom_losses.categorical_focal_loss_with_iou(alpha=0.50, gamma=1.25, model=model)],
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if not checkpoints_path is None:
        open(checkpoints_path + "_config.json", "w").write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)
        if validate:
            print("Verifying val dataset")
            verify_segmentation_dataset(val_images, val_annotations, n_classes)

    train_gen = image_segmentation_generator(train_images, train_annotations, batch_size, n_classes, input_height,
                                             input_width, output_height, output_width)

    if validate:
        val_gen = image_segmentation_generator(val_images, val_annotations, val_batch_size, n_classes, input_height,
                                               input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)


def train_i3d_inception(model,
                        train_features_folder,
                        train_annotations_folder,
                        input_height=None,
                        input_width=None,
                        n_classes=None,
                        verify_dataset=True,
                        checkpoints_path=None,
                        epochs=5,
                        batch_size=2,
                        validate=False,
                        val_images=None,
                        val_annotations=None,
                        val_batch_size=2,
                        auto_resume_checkpoint=False,
                        load_weights=None,
                        steps_per_epoch=512,
                        optimizer_name='adam'
                        ):
    if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
        # create the model from the name
        assert (not n_classes is None), "Please provide the n_classes"
        if (not input_height is None) and (not input_width is None):
            model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert not (val_images is None)
        assert not (val_annotations is None)

    if not optimizer_name is None:
        # model.compile(loss='categorical_crossentropy',
        # 	optimizer= optimizer_name ,
        # 	metrics=['accuracy'])

        # model.compile(loss=[custom_losses.categorical_focal_loss_with_iou(alpha=0.50, gamma=1.25, model=model)],
        #               optimizer=optimizer_name,
        #               metrics=['accuracy', iou_metric_wrapper(output_height, output_width, n_classes)])

        model.compile(loss=[custom_losses.categorical_focal_loss_with_iou(alpha=0.50, gamma=1.25, model=model)],
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if not checkpoints_path is None:
        open(checkpoints_path + "_config.json", "w").write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_features_folder, train_annotations_folder, n_classes)
        if validate:
            print("Verifying val dataset")
            verify_segmentation_dataset(train_features_folder, train_annotations_folder, n_classes)

    train_gen = image_segmentation_generator_i3d_inception(train_features_folder, train_annotations_folder, batch_size,
                                                           n_classes, input_height, input_width, output_height,
                                                           output_width)

    if validate:
        val_gen = image_segmentation_generator_i3d_inception(val_images, val_annotations, val_batch_size, n_classes,
                                                             input_height, input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)


def train_with_weighted_output(model,
                               train_images,
                               train_annotations,
                               input_height=None,
                               input_width=None,
                               n_classes=None,
                               verify_dataset=True,
                               checkpoints_path=None,
                               epochs=5,
                               batch_size=2,
                               validate=False,
                               val_images=None,
                               val_annotations=None,
                               val_batch_size=2,
                               auto_resume_checkpoint=False,
                               load_weights=None,
                               steps_per_epoch=512,
                               optimizer_name='adam'
                               ):
    if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
        # create the model from the name
        assert (not n_classes is None), "Please provide the n_classes"
        if (not input_height is None) and (not input_width is None):
            model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert not (val_images is None)
        assert not (val_annotations is None)

    if not optimizer_name is None:
        # model.compile(loss='categorical_crossentropy',
        # 	optimizer= optimizer_name ,
        # 	metrics=['accuracy'])

        # model.compile(loss=[custom_losses.categorical_focal_loss_with_iou(alpha=0.50, gamma=1.25, model=model)],
        #               optimizer=optimizer_name,
        #               metrics=['accuracy', iou_metric_wrapper(output_height, output_width, n_classes)])

        model.compile(loss={
            "main_output_activation": custom_losses.categorical_focal_loss_with_iou(alpha=0.50, gamma=1.25,
                                                                                    model=model),
            "second_output_activation": smooth_l1_loss,
        },
            optimizer=optimizer_name,
            metrics=['accuracy'])

    if not checkpoints_path is None:
        open(checkpoints_path + "_config.json", "w").write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)
        if validate:
            print("Verifying val dataset")
            verify_segmentation_dataset(val_images, val_annotations, n_classes)

    train_gen = image_segmentation_generator_with_weighted_output(train_images, train_annotations, batch_size, n_classes, input_height,
                                             input_width, output_height, output_width)

    if validate:
        val_gen = image_segmentation_generator_with_weighted_output(val_images, val_annotations, val_batch_size, n_classes, input_height,
                                               input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)


def train_temporal_with_weighted_output(model,
                                        train_images,
                                        train_annotations,
                                        input_height=None,
                                        input_width=None,
                                        n_classes=None,
                                        verify_dataset=True,
                                        checkpoints_path=None,
                                        epochs=5,
                                        batch_size=2,
                                        validate=False,
                                        val_images=None,
                                        val_annotations=None,
                                        val_batch_size=2,
                                        auto_resume_checkpoint=False,
                                        load_weights=None,
                                        steps_per_epoch=512,
                                        optimizer_name='adam'
                                        ):
    if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
        # create the model from the name
        assert (not n_classes is None), "Please provide the n_classes"
        if (not input_height is None) and (not input_width is None):
            model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert not (val_images is None)
        assert not (val_annotations is None)

    if not optimizer_name is None:
        # model.compile(loss='categorical_crossentropy',
        # 	optimizer= optimizer_name ,
        # 	metrics=['accuracy'])

        # model.compile(loss=[custom_losses.categorical_focal_loss_with_iou(alpha=0.50, gamma=1.25, model=model)],
        #               optimizer=optimizer_name,
        #               metrics=['accuracy', iou_metric_wrapper(output_height, output_width, n_classes)])

        model.compile(loss={
            "main_output_activation": custom_losses.categorical_focal_loss_with_iou(alpha=0.50, gamma=1.25,
                                                                                    model=model),
            "second_output_activation": smooth_l1_loss,
        },
            optimizer=optimizer_name,
            metrics=['accuracy'])

    if not checkpoints_path is None:
        open(checkpoints_path + "_config.json", "w").write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)
        if validate:
            print("Verifying val dataset")
            verify_segmentation_dataset(val_images, val_annotations, n_classes)

    train_gen = image_segmentation_temporal_generator_with_weighted_output(train_images, train_annotations, batch_size,
                                                                           n_classes, input_height,
                                                                           input_width, output_height, output_width)

    if validate:
        val_gen = image_segmentation_temporal_generator_with_weighted_output(val_images, val_annotations,
                                                                             val_batch_size,
                                                                             n_classes, input_height,
                                                                             input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)


def train_bounding_box_based_network(model, train_images, train_annotations, input_height=None, input_width=None,
                                     n_classes=None, verify_dataset=True, checkpoints_path=None, epochs=5, batch_size=2,
                                     validate=False, val_images=None, val_annotations=None, val_batch_size=2,
                                     auto_resume_checkpoint=False, load_weights=None, steps_per_epoch=512,
                                     optimizer_name='adam', optimizer_lr=0.001, optimizer_decay=0.001):
    if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
        # create the model from the name
        assert (not n_classes is None), "Please provide the n_classes"
        if (not input_height is None) and (not input_width is None):
            model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    # output_height = model.output_height
    # output_width = model.output_width
    output_height = input_height
    output_width = input_width

    if validate:
        assert not (val_images is None)
        assert not (val_annotations is None)

    if optimizer_name is not None:
        # model.compile(loss='categorical_crossentropy',
        # 	optimizer= optimizer_name ,
        # 	metrics=['accuracy'])

        adam = optimizers.Adam(lr=optimizer_lr, decay=optimizer_decay)

        model.compile(loss=bounding_box_based_network_loss_gpu,
                      optimizer=adam,
                      metrics=['accuracy'])

    if checkpoints_path is not None:
        open(checkpoints_path + "_config.json", "w").write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)
        if validate:
            print("Verifying val dataset")
            verify_segmentation_dataset(val_images, val_annotations, n_classes)

    train_gen = image_segmentation_generator_bounding_box_based_network(train_images, train_annotations, batch_size, n_classes, input_height,
                                             input_width, output_height, output_width)

    if validate:
        val_gen = image_segmentation_generator_bounding_box_based_network(val_images, val_annotations, val_batch_size, n_classes, input_height,
                                               input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)


def train_bounding_box_iou_based_network(model, train_images, train_annotations, input_height=None, input_width=None,
                                         n_classes=None, verify_dataset=True, checkpoints_path=None, epochs=5,
                                         batch_size=2,
                                         validate=False, val_images=None, val_annotations=None, val_batch_size=2,
                                         auto_resume_checkpoint=False, load_weights=None, steps_per_epoch=512,
                                         optimizer_name='adam', optimizer_lr=0.001, optimizer_decay=0.001):
    if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
        # create the model from the name
        assert (not n_classes is None), "Please provide the n_classes"
        if (not input_height is None) and (not input_width is None):
            model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    # output_height = model.output_height
    # output_width = model.output_width
    output_height = input_height
    output_width = input_width

    if validate:
        assert not (val_images is None)
        assert not (val_annotations is None)

    if optimizer_name is not None:
        # model.compile(loss='categorical_crossentropy',
        # 	optimizer= optimizer_name ,
        # 	metrics=['accuracy'])

        adam = optimizers.Adam(lr=optimizer_lr, decay=optimizer_decay)

        model.compile(loss=bounding_box_iou_based_network_loss,
                      optimizer=adam,
                      metrics=['accuracy', bounding_box_iou_based_network_metric])

    if checkpoints_path is not None:
        open(checkpoints_path + "_config.json", "w").write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)
        if validate:
            print("Verifying val dataset")
            verify_segmentation_dataset(val_images, val_annotations, n_classes)

    train_gen = image_segmentation_generator_bounding_box_iou_based_network(train_images, train_annotations, batch_size,
                                                                            n_classes, input_height,
                                                                            input_width, output_height, output_width)

    if validate:
        val_gen = image_segmentation_generator_bounding_box_iou_based_network(val_images, val_annotations,
                                                                              val_batch_size,
                                                                              n_classes, input_height,
                                                                              input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)


def train_IoU_network(model,
                      train_images,
                      train_annotations,
                      input_height=None,
                      input_width=None,
                      n_classes=None,
                      verify_dataset=True,
                      checkpoints_path=None,
                      epochs=5,
                      batch_size=2,
                      validate=False,
                      val_images=None,
                      val_annotations=None,
                      val_batch_size=2,
                      auto_resume_checkpoint=False,
                      load_weights=None,
                      steps_per_epoch=512,
                      optimizer_name='adam'
                      ):
    if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
        # create the model from the name
        assert (not n_classes is None), "Please provide the n_classes"
        if (not input_height is None) and (not input_width is None):
            model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    # output_height = model.output_height
    # output_width = model.output_width
    output_height = None
    output_width = None

    if validate:
        assert not (val_images is None)
        assert not (val_annotations is None)

    if not optimizer_name is None:
        # model.compile(loss='categorical_crossentropy',
        # 	optimizer= optimizer_name ,
        # 	metrics=['accuracy'])

        model.compile(loss=custom_losses.smooth_l1_loss,
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if not checkpoints_path is None:
        open(checkpoints_path + "_config.json", "w").write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)
        if validate:
            print("Verifying val dataset")
            verify_segmentation_dataset(val_images, val_annotations, n_classes)

    train_gen = IoU_network_image_segmentation_generator(train_images, train_annotations, batch_size, n_classes,
                                                         input_height,
                                                         input_width, output_height, output_width)

    if validate:
        val_gen = IoU_network_image_segmentation_generator(val_images, val_annotations, val_batch_size, n_classes,
                                                           input_height,
                                                           input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)


def train_i3d(model,
              train_images,
              train_annotations,
              input_height=None,
              input_width=None,
              n_classes=None,
              verify_dataset=True,
              checkpoints_path=None,
              epochs=5,
              batch_size=2,
              validate=False,
              val_images=None,
              val_annotations=None,
              val_batch_size=2,
              auto_resume_checkpoint=False,
              load_weights=None,
              steps_per_epoch=512,
              optimizer_name='adam',
              lr_custom=0.001,
              lr_decay=0.0
              ):
    if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
        # create the model from the name
        assert (not n_classes is None), "Please provide the n_classes"
        if (not input_height is None) and (not input_width is None):
            model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert not (val_images is None)
        assert not (val_annotations is None)

    if not optimizer_name is None:
        # model.compile(loss='categorical_crossentropy',
        # 	optimizer= optimizer_name ,
        # 	metrics=['accuracy'])

        adam = optimizers.Adam(lr=lr_custom, beta_1=0.9, beta_2=0.999, decay=lr_decay)
        model.compile(loss={
            "main_output_activation": custom_losses.categorical_focal_loss_with_iou(alpha=0.50, gamma=1.25,
                                                                                    model=model),
            "second_output_activation": smooth_l1_loss,
        },
            optimizer=adam,
            metrics=['accuracy'])

    if not checkpoints_path is None:
        open(checkpoints_path + "_config.json", "w").write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)
        if validate:
            print("Verifying val dataset")
            verify_segmentation_dataset(val_images, val_annotations, n_classes)

    train_gen = image_segmentation_generator_i3d(train_images, train_annotations, batch_size, n_classes, input_height,
                                                 input_width, output_height, output_width)

    if validate:
        val_gen = image_segmentation_generator_i3d(val_images, val_annotations, val_batch_size, n_classes, input_height,
                                                   input_width, output_height, output_width)

    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            # model.fit_generator(train_gen, steps_per_epoch, epochs=1, callbacks=[reduce_lr])
            model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            # model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1,
            #                     callbacks=[reduce_lr])
            model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)


def train_two_stream(model,
                     train_images,
                     train_flows,
                     train_annotations,
                     input_height=None,
                     input_width=None,
                     n_classes=None,
                     verify_dataset=True,
                     checkpoints_path=None,
                     epochs=5,
                     batch_size=2,
                     validate=False,
                     val_images=None,
                     val_flows=None,
                     val_annotations=None,
                     val_batch_size=2,
                     auto_resume_checkpoint=False,
                     load_weights=None,
                     steps_per_epoch=512,
                     optimizer_name='adam'
                     ):
    if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
        # create the model from the name
        assert (not n_classes is None), "Please provide the n_classes"
        if (not input_height is None) and (not input_width is None):
            model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert not (val_images is None)
        assert not (val_flows is None)
        assert not (val_annotations is None)

    if not optimizer_name is None:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if not checkpoints_path is None:
        open(checkpoints_path + "_config.json", "w").write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying train dataset")
        two_stream_verify_segmentation_dataset(train_images, train_flows, train_annotations, n_classes)
        if validate:
            print("Verifying val dataset")
            two_stream_verify_segmentation_dataset(val_images, train_flows, val_annotations, n_classes)

    train_gen = two_stream_image_segmentation_generator(train_images, train_flows, train_annotations, batch_size,
                                                        n_classes, input_height, input_width, output_height,
                                                        output_width)

    if validate:
        val_gen = two_stream_image_segmentation_generator(val_images, val_flows, val_annotations, val_batch_size,
                                                          n_classes, input_height, input_width, output_height,
                                                          output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))

                ## replace_previous_checkpoint_with_empty_file(checkpoints_path, ep)
            print("Finished Epoch", ep)
