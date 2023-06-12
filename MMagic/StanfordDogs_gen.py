_base_ = [
    '/mnt/sda/2022-0526/home/scc/ytx/Software/mmagic/configs/biggan/biggan-deep_cvt-hugging-face_rgb_imagenet1k-256x256.py',
]

dataset_type = 'StanfordDogDataset'

dog_classes = ('n02105056-groenendael', 'n02099601-golden_retriever', 'n02113712-miniature_poodle', 'n02086910-papillon', 'n02093991-Irish_terrier', 'n02098413-Lhasa', 'n02100583-vizsla', 'n02095889-Sealyham_terrier', 'n02092339-Weimaraner', 'n02100877-Irish_setter', 'n02106550-Rottweiler', 'n02094433-Yorkshire_terrier', 'n02097209-standard_schnauzer', 'n02102040-English_springer', 'n02107908-Appenzeller', 'n02111889-Samoyed', 'n02108000-EntleBucher', 'n02097658-silky_terrier', 'n02091032-Italian_greyhound', 'n02096294-Australian_terrier', 'n02091467-Norwegian_elkhound', 'n02106382-Bouvier_des_Flandres', 'n02101556-clumber', 'n02093428-American_Staffordshire_terrier', 'n02105251-briard', 'n02113186-Cardigan', 'n02106662-German_shepherd', 'n02085782-Japanese_spaniel', 'n02102480-Sussex_spaniel', 'n02093754-Border_terrier', 'n02105855-Shetland_sheepdog', 'n02087046-toy_terrier', 'n02096437-Dandie_Dinmont', 'n02113624-toy_poodle', 'n02097298-Scotch_terrier', 'n02107142-Doberman', 'n02085936-Maltese_dog', 'n02093647-Bedlington_terrier', 'n02113799-standard_poodle', 'n02100735-English_setter', 'n02096051-Airedale', 'n02108551-Tibetan_mastiff', 'n02091134-whippet', 'n02111500-Great_Pyrenees', 'n02110185-Siberian_husky', 'n02109047-Great_Dane', 'n02112350-keeshond', 'n02110806-basenji', 'n02097047-miniature_schnauzer', 'n02115913-dhole', 'n02104365-schipperke', 'n02095570-Lakeland_terrier', 'n02091635-otterhound', 'n02105505-komondor', 'n02089078-black-and-tan_coonhound', 'n02102973-Irish_water_spaniel', 'n02090379-redbone', 'n02105412-kelpie', 'n02091831-Saluki', 'n02085620-Chihuahua', 'n02107312-miniature_pinscher', 'n02088364-beagle', 'n02102318-cocker_spaniel', 'n02111277-Newfoundland', 'n02095314-wire-haired_fox_terrier', 'n02097474-Tibetan_terrier', 'n02101388-Brittany_spaniel', 'n02099849-Chesapeake_Bay_retriever', 'n02113978-Mexican_hairless', 'n02107683-Bernese_mountain_dog', 'n02106030-collie', 'n02108089-boxer', 'n02086240-Shih-Tzu', 'n02096585-Boston_bull', 'n02102177-Welsh_springer_spaniel', 'n02093256-Staffordshire_bullterrier', 'n02094258-Norwich_terrier', 'n02106166-Border_collie', 'n02097130-giant_schnauzer', 'n02088094-Afghan_hound', 'n02099429-curly-coated_retriever', 'n02110063-malamute', 'n02112706-Brabancon_griffon', 'n02107574-Greater_Swiss_Mountain_dog', 'n02088466-bloodhound', 'n02086646-Blenheim_spaniel', 'n02098105-soft-coated_wheaten_terrier', 'n02105162-malinois', 'n02090721-Irish_wolfhound', 'n02091244-Ibizan_hound', 'n02093859-Kerry_blue_terrier', 'n02087394-Rhodesian_ridgeback', 'n02108915-French_bulldog', 'n02098286-West_Highland_white_terrier', 'n02116738-African_hunting_dog', 'n02089867-Walker_hound', 'n02104029-kuvasz', 'n02099712-Labrador_retriever', 'n02088632-bluetick', 'n02111129-Leonberg', 'n02099267-flat-coated_retriever', 'n02112018-Pomeranian', 'n02086079-Pekinese', 'n02115641-dingo', 'n02113023-Pembroke', 'n02101006-Gordon_setter', 'n02110958-pug', 'n02088238-basset', 'n02109961-Eskimo_dog', 'n02110627-affenpinscher', 'n02108422-bull_mastiff', 'n02092002-Scottish_deerhound', 'n02100236-German_short-haired_pointer', 'n02089973-English_foxhound', 'n02096177-cairn', 'n02109525-Saint_Bernard', 'n02105641-Old_English_sheepdog', 'n02112137-chow', 'n02094114-Norfolk_terrier', 'n02090622-borzoi')


metainfo = {
    'classes': dog_classes,
}

train_pipeline = [
    dict(type='LoadImageFromFile',  # Load images from files
        key='gt',  # Keys in results to find the corresponding path
        color_type='color',  # Color type of image
        channel_order='rgb',  # Channel order of image
        imdecode_backend='cv2'),  # decode backend
    dict(type='Resize', keys=['gt'], scale=(256, 256)),
    dict(type='Normalize', keys=['gt'], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], to_rgb=True),
    # dict(type='ToTensor', keys=['gt']),
    # dict(type='ImageToTensor', keys=['gt']),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    _delete_ = True,
    num_workers=8, 
    persistent_workers=False,  # Whether maintain the workers Dataset instances alive
    sampler=dict(type='InfiniteSampler', shuffle=True),  # The type of data sampler
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root='/mnt/sda/2022-0526/home/scc/ytx/ANN/Main_Project/data/StanfordDogs',
        ann_file='squareCroppedImages_train_annotation.txt',
        data_prefix="/mnt/sda/2022-0526/home/scc/ytx/ANN/Main_Project/data/StanfordDogs",
        pipeline=train_pipeline))

val_dataloader = dict(
    _delete_ = True,
    num_workers=8, 
    persistent_workers=False,  # Whether maintain the workers Dataset instances alive
    sampler=dict(type='InfiniteSampler', shuffle=True),  # The type of data sampler
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root='/mnt/sda/2022-0526/home/scc/ytx/ANN/Main_Project/data/StanfordDogs',
        ann_file='squareCroppedImages_test_annotation.txt',
        data_prefix="/mnt/sda/2022-0526/home/scc/ytx/ANN/Main_Project/data/StanfordDogs",
        pipeline=train_pipeline))

test_dataloader = val_dataloader

train_cfg = dict(
    _delete_ = True,
    type='IterBasedTrainLoop',  # The name of train loop type
    max_iters=10000,  # The number of total iterations
    val_interval=5000,  # The number of validation interval iterations
)


num_classes=120
model = dict(
    num_classes=num_classes,
    generator=dict(
        num_classes=num_classes,),
    discriminator=dict(
        num_classes=num_classes,))

# optim_wrapper = dict(
#     _delete_ = True,
#     discriminator=dict(
#         _delete_ = True,
#         type='OptimWrapper',
#         optimizer=dict(type='Adam', lr=0.00001),
#     )
# )

optim_wrapper = dict(
    _delete_ = True,
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        optimizer=dict(type='Adam', lr=0.0016, betas=(0, 0.9919919678228657))),
    discriminator=dict(
        optimizer=dict(
            type='Adam',
            lr=0.0018823529411764706,
            betas=(0, 0.9905854573074332))))

param_scheduler = dict(  # Config of learning policy
    type='MultiStepLR', by_epoch=False, milestones=[200000], gamma=0.5
)
work_dir = "/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project/work_dirs/task3/dog"
# default_hooks = dict(visualization=dict(type='DetVisualizationHook', draw=True))
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)

# VIS_HOOK
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=500,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]