import os
import pandas as pd

SM_num_cols = ['Age', 'TimeTin-months', 'MeanLoudness', 'MinLoudness', 'MaxLoudness', 'CurrLoudness', 'MeanAnnoyancePastWeek',
           'MinAnnoyanceGoodDay', 'MaxAnnoyanceBadDay', 'CurrAnnoyance', 'AwarenessOpenEnded', 'HearingLoss_Echelle',
               "Fleeting tinnitus"
               ]
SM_already_categorical=['Female', 'TinType_A pure tone / single pitch', 'TinType_A mixture of tones', 'TinType_A low buzzing',
           'TinType_A high buzzing', 'TinType_Electric / interference type sound', 'TinType_A low rumbling',
           'TinType_A static noise', 'TinType_Clicking', 'TinType_Intermittent beeping (morse code)',
           'TinType_A whooshing noise - pulsatile', 'TinType_A whooshing noise - not in time with the pulse',
           'TinType_A pulsating sound', 'TinType_Other', 'cause_Noise trauma', 'cause_Noise induced hearing loss',
           'cause_ageRelated', 'cause_Sudden hearing loss', "cause_Meniere's",  'cause_Head or neck injury',
                        'cause_Barotrauma - air / water pressure',
            'cause_TMJ (issues with the jaw)', 'cause_Psychological (stress, anxiety, depression)',
            'cause_Ototoxic (from drugs or medication)', 'cause_Otosclerosis', 'cause_Eustachian tube dysfunction',
                        'cause_Dental treatment', 'cause_Allergy to something',
            'cause_Ear wax procedure (syringing, candling or other related procedure)',
            'cause_Metabolic (diabetes, thyroid, B12, hyperlipidaemia etc.)',
           'cause_Virus (ear infection, flu, cold)', 'cause_Ear wax build up',
            'cause_Unknown',
                        "Fluctuaction_categ", 'InflStress', 'InflAnxiety',
                        'InflGoodSleep', 'InflBadSleep',
             'ReactSound_Worse', 'ReactSound_Better', "Mask_category",
            'InflIntenseworkout', 'InflModerateWorkout', 'InflLightEx',
             'TinChange_Pressing the jaw on the side', 'TinChange_Pushing the jaw backwards',
           'TinChange_Pushing the lower jaw outwards rapidly',
           'TinChange_Pushing your hand against your forehead while resisting with the neck muscles',
           'TinChange_Clenching the teeth together', 'TinChange_TiltingÂ\xa0your head backwards',
           'TinChange_No change with any movements', 'JawPain_Yes, my jaw sometimes feels painful',
            'JawPain_Yes, I struggle to fully move my jaw', 'JawPain_Yes, my jaw sometimes feels tired',
           'JawPain_Yes, muscles around my jaw feel tight or tense',
           'JawPain_Yes, I have several popping and clicking noises in my jaw',
           'JawPain_Yes, I have been diagnosed with TMJ dysfunction', 'JawPain_No real issues that I am aware of',
            'Grinding',
           'Clenching',
            'NeckStiffness1', 'NeckStiffness2', 'NeckStiffness3', 'NeckStiffness4',
           'NeckStiffness5', 'NeckStiffness0', 'Headache_YesorNo', 'Headache_Neck', 'Headache_Jaw', 'EarFullness6',
           'EarFullness5', 'EarFullness4', 'EarFullness3', 'EarFullness2', 'EarFullness1', 'EarFullness0'

           ]


JS_num_cols = ['age', 'tschq-months_onset','tschq-tinpitch',"tinnitus_laterality", 'tschq-personal_loudness', 'thi_score', 'tq_score', 'ts_1', 'ts_2', 'ts_3', 'ts_4', 'ts_5',
               'ts_6', 'tschq-awareness_time', 'tschq-angerness_time',
                'whoqol_1', 'whoqol_2', 'whoqol_3', 'whoqol_4',
                            'tschq-hyperacusis' ]
JS_categorical_essential = [ 'sex', 'tschq-handedness', 'tschq-hearproblems',  'tschq-tintype', 'tschq-pulsating',
                             'tschq-tinside', 'tschq-familiy', 'tschq-inital-perception',
                            "tschq-cause_loud_blast", "tschq-cause_stress",
                             "tschq-cause_whiplash", "tschq-cause_head_trauma",
                            "tschq-cause_change_in_hearing", "tschq-cause_other",
                             'tschq-fluctuations',
                           'tschq-intermitent', 'tschq-stress', 'tschq-sleep', 'tschq-sounds-worsen-tinnitus',
                             'tschq-sounds-suppress-tinnitus','tschq-noise-pain', 'tschq-somatic',
                           'tschq-tmj','tschq-neckpain', 'tschq-headache',  'tschq-vertigo',
                           'tschq-n-treatments', 'tschq-psychiatry', 'tschq-ha'
                              ]
JS_already_categorical =  [ 'mdi_1', 'mdi_2', 'mdi_3', 'mdi_4', 'mdi_5', 'mdi_6', 'mdi_7', 'mdi_8a',
                           'mdi_8b', 'mdi_9', 'mdi_10a', 'mdi_10b',  'tfi_1', 'tfi_2', 'tfi_3', 'tfi_4',
                           'tfi_5', 'tfi_6', 'tfi_7', 'tfi_8', 'tfi_9', 'tfi_10', 'tfi_11', 'tfi_12', 'tfi_13',
                           'tfi_14', 'tfi_15', 'tfi_16', 'tfi_17', 'tfi_18', 'tfi_19', 'tfi_20', 'tfi_21', 'tfi_22',
                           'tfi_23', 'tfi_24', 'tfi_25', 'thi_1', 'thi_2', 'thi_3', 'thi_4', 'thi_5', 'thi_6', 'thi_7',
                           'thi_8', 'thi_9', 'thi_10', 'thi_11', 'thi_12', 'thi_13', 'thi_14', 'thi_15', 'thi_16',
                           'thi_17', 'thi_18', 'thi_19', 'thi_20', 'thi_21', 'thi_22', 'thi_23', 'thi_24', 'thi_25',
                        'tq_1', 'tq_2', 'tq_3', 'tq_4', 'tq_5', 'tq_6', 'tq_7', 'tq_8', 'tq_9', 'tq_10',
                           'tq_11', 'tq_12', 'tq_13', 'tq_14', 'tq_15', 'tq_16', 'tq_17', 'tq_18', 'tq_19', 'tq_20',
                           'tq_21', 'tq_22', 'tq_23', 'tq_24', 'tq_25', 'tq_26', 'tq_27', 'tq_28', 'tq_29', 'tq_30',
                           'tq_31', 'tq_32', 'tq_33', 'tq_34', 'tq_35', 'tq_36', 'tq_37', 'tq_38', 'tq_39', 'tq_40',
                           'tq_41', 'tq_42', 'tq_43', 'tq_44', 'tq_45', 'tq_46', 'tq_47', 'tq_48', 'tq_49', 'tq_50',
                           'tq_51', 'tq_52']

merge_num=["TimeTin-months","Age",	"MeanLoudness",	"MeanAnnoyancePastWeek","AwarenessOpenEnded"
]
merge_categ=["Female","cause_head_or_neck_injury", "cause_Psychological (stress, anxiety, depression)", "cause_Noise trauma",
             "Cause_change_hearing",
             "ReactSound_Worse",	"HearingLoss",	"pain_cervical", "Headache_YesorNo",	"Tin_somato",	"TMJ_problem",
             "TinType_A pure tone / single pitch", "TinType_A whooshing noise - not in time with the pulse", "tin_puls",
                "InflStress", "Tin_masked"
             ]


SM_str_fields = {'Age':['Age (years)', {}], 'TimeTin-months':['Tinnitus duration (months)', {}],
                 'MeanLoudness':['Average loudness', {}], 'MinLoudness':['Loudness on good days', {}],
                 'MaxLoudness':['Loudness on bad days', {}], 'CurrLoudness':['Current loudness', {}],
                 'MeanAnnoyancePastWeek':['Average annoyance', {}], 'MinAnnoyanceGoodDay':['Annoyance on good days', {}]
                    , 'MaxAnnoyanceBadDay':['Annoyance on bad days', {}], 'CurrAnnoyance':['Current annoyance', {}],
                 'AwarenessOpenEnded':['% of time aware of tinnitus', {}],
                 'HearingLoss_Echelle':['Hearing loss grade (0 : none, 3 : severe) ', {}],
               "Fleeting tinnitus":['Frequency of fleeting tinnitus (0 : never, 5 : daily)', {}],

                 'Female':['Gender', {0:"Male", 1:"Female"}],
                 'TinType_A pure tone / single pitch':['Tinnitus sound : A pure tone', {0:"No", 1:"Yes"}],
             'TinType_A mixture of tones':['Tinnitus sound : A mixture of tones', {0:"No", 1:"Yes"}],
                 'TinType_A low buzzing':['Tinnitus sound : A low buzzing', {0:"No", 1:"Yes"}],
           'TinType_A high buzzing':['Tinnitus sound : A high buzzing', {0:"No", 1:"Yes"}],
                 'TinType_Electric / interference type sound':['Tinnitus sound : electric/interference', {0:"No", 1:"Yes"}],
                 'TinType_A low rumbling':['Tinnitus sound : A low rumbling', {0:"No", 1:"Yes"}],
           'TinType_A static noise':['Tinnitus sound : A static noise', {0:"No", 1:"Yes"}],
                 'TinType_Clicking':['Tinnitus sound : clicking', {0:"No", 1:"Yes"}],
                 'TinType_Intermittent beeping (morse code)':['Tinnitus sound : beeping (morse code)', {0:"No", 1:"Yes"}],
           'TinType_A whooshing noise - pulsatile':['Tinnitus sound : A pulsatile whooshing noise', {0:"No", 1:"Yes"}],
                 'TinType_A whooshing noise - not in time with the pulse':['Tinnitus sound : A non-pulsatile whooshing noise', {0:"No", 1:"Yes"}],
           'TinType_A pulsating sound':['Tinnitus sound : pulsatile sound', {0:"No", 1:"Yes"}],
                 'TinType_Other':['Tinnitus sound : other', {0:"No", 1:"Yes"}],
                 'cause_Noise trauma':['Tinnitus cause : noise trauma', {0:"No", 1:"Yes"}],
                 'cause_Noise induced hearing loss':['Tinnitus cause : hearing loss', {0:"No", 1:"Yes"}],
           'cause_ageRelated':['Tinnitus cause : age-related hearing loss', {0:"No", 1:"Yes"}],
                'cause_Sudden hearing loss':['Tinnitus cause : sudden hearing loss', {0:"No", 1:"Yes"}],
                 "cause_Meniere's":['Tinnitus cause : Meniere disease', {0:"No", 1:"Yes"}],
                 'cause_Head or neck injury':['Tinnitus cause : head or neck injury', {0:"No", 1:"Yes"}],
                        'cause_Barotrauma - air / water pressure':['Tinnitus cause : barotrauma', {0:"No", 1:"Yes"}],
            'cause_TMJ (issues with the jaw)':['Tinnitus cause : TMJ dysfunction', {0:"No", 1:"Yes"}],
            'cause_Psychological (stress, anxiety, depression)':['Tinnitus cause : psychological (stress, anxiety, depression)', {0:"No", 1:"Yes"}],
            'cause_Ototoxic (from drugs or medication)':['Tinnitus cause : ototoxicity', {0:"No", 1:"Yes"}],
            'cause_Otosclerosis':['Tinnitus cause : otosclerosis', {0:"No", 1:"Yes"}],
                 'cause_Eustachian tube dysfunction':['Tinnitus cause : eustachian tube dysfunction', {0:"No", 1:"Yes"}],
                        'cause_Dental treatment':['Tinnitus cause : dental treatment', {0:"No", 1:"Yes"}],
                 'cause_Allergy to something':['Tinnitus cause : allergy', {0:"No", 1:"Yes"}],
            'cause_Ear wax procedure (syringing, candling or other related procedure)':['Tinnitus cause : ear wax procedure (syringing, candling...)', {0:"No", 1:"Yes"}],
            'cause_Metabolic (diabetes, thyroid, B12, hyperlipidaemia etc.)':['Tinnitus cause : metabolic (diabetes, thyroid, B12, hyperlipidaemia etc.)', {0:"No", 1:"Yes"}],
           'cause_Virus (ear infection, flu, cold)':['Tinnitus cause : virus or infection', {0:"No", 1:"Yes"}],
                 'cause_Ear wax build up':['Tinnitus cause : ear wax build up', {0:"No", 1:"Yes"}],
            'cause_Unknown':['Tinnitus cause : unknown', {0:"No", 1:"Yes"}],
                        "Fluctuaction_categ":['Fluctuations of tinnitus', {0:"No fluctuations", 1:"Grows louder as day progresses",
                                                                           2:"Grows quieter as day progresses",
                                                                           3: "Changes within the day or over days without any pattern"}],
             'InflStress':['Influence of stress over tinnitus', {-1 : "Worsens", 0:"No effect", 1:"Improves"}],
                 'InflAnxiety':['Influence of anxiety over tinnitus', {-1 : "Worsens", 0:"No effect", 1:"Improves"}],
                    'InflGoodSleep':['Influence of a good night sleep over tinnitus', {-1 : "Worsens", 0:"No effect", 1:"Improves"}],
                 'InflBadSleep':['Influence of poor sleep over tinnitus', {-1 : "Worsens", 0:"No effect", 1:"Improves"}],
             'ReactSound_Worse':['Some sounds can worsen tinnitus', {0:"No", 1:"Yes"}],
                 'ReactSound_Better':['Some sounds can reduce tinnitus', {0:"No", 1:"Yes"}],
                 "Mask_category":['Tinnitus sound masking', {0:"No masking", 1:"Only a small selection of specific sounds",
                                                                           2:"Shower / water noises",
                                                                           3: "TV, Music or general background noise",
                                                             4: "White noise or special masking noises",
                                                             5: "Masked by nearly all sounds"}],
            'InflIntenseworkout':['Influence of intense workout over tinnitus', {-1 : "Worsens", 0:"No effect", 1:"Improves"}],
             'InflModerateWorkout':['Influence of moderate workout over tinnitus', {-1 : "Worsens", 0:"No effect", 1:"Improves"}],
             'InflLightEx':['Influence of light exercise over tinnitus', {-1 : "Worsens", 0:"No effect", 1:"Improves"}],
            'TinChange_Clenching the teeth together':['Somatosensory change : clenching teeth', {0:"No", 1:"Yes"}],
             'TinChange_Pressing the jaw on the side':['Somatosensory change : pressing the jaw on the side', {0:"No", 1:"Yes"}],
                'TinChange_Pushing the jaw backwards':['Somatosensory change : pushing jaw backwards', {0:"No", 1:"Yes"}],
           'TinChange_Pushing the lower jaw outwards rapidly':['Somatosensory change : Pushing the jaw outwards rapidly', {0:"No", 1:"Yes"}],
           'TinChange_Pushing your hand against your forehead while resisting with the neck muscles':
                     ['Somatosensory change : Pushing your hand against your forehead while resisting with the neck muscles', {0:"No", 1:"Yes"}],

            'TinChange_TiltingÂ\xa0your head backwards':
                     ['Somatosensory change : Tilting your head backwards', {0:"No", 1:"Yes"}],
           'TinChange_No change with any movements':
                     ['Somatosensory change : no change with any of these actions', {0:"No", 1:"Yes"}],
            'JawPain_Yes, my jaw sometimes feels painful':
                     ['Jaw : my jaw sometimes feels painful', {0:"No", 1:"Yes"}],
            'JawPain_Yes, I struggle to fully move my jaw':
                     ['Jaw : I struggle to fully move my jaw', {0:"No", 1:"Yes"}],
                 'JawPain_Yes, my jaw sometimes feels tired':
                     ['Jaw : I struggle to fully move my jaw', {0:"No", 1:"Yes"}],
           'JawPain_Yes, muscles around my jaw feel tight or tense':
                     ['Jaw : muscles around my jaw feel tight or tense', {0:"No", 1:"Yes"}],
           'JawPain_Yes, I have several popping and clicking noises in my jaw':
                     ['Jaw : I have several popping and clicking noises in my jaw', {0:"No", 1:"Yes"}],
           'JawPain_Yes, I have been diagnosed with TMJ dysfunction':
                     ['Jaw : I have been diagnosed with TMJ dysfunction', {0:"No", 1:"Yes"}],
            'JawPain_No real issues that I am aware of':
                     ['Jaw : No real issues that I am aware of', {0:"No", 1:"Yes"}],
            'Grinding':
                     ['Bruxism : I grind my teeth during my sleep', {0:"No", 1:"Yes"}],
           'Clenching':
                     ['Bruxism : I often clench my teeth without realising it', {0:"No", 1:"Yes"}],
            'NeckStiffness1':
                     ['Neck stiffness : after certain physical activity', {0:"No", 1:"Yes"}],
            'NeckStiffness2':
                     ['Neck stiffness : from bad posture', {0:"No", 1:"Yes"}],
            'NeckStiffness3':
                     ['Neck stiffness : I have an associated medical condition', {0:"No", 1:"Yes"}],
            'NeckStiffness4':
                     ['Neck stiffness : from lying in bed / sleeping', {0:"No", 1:"Yes"}],
            'NeckStiffness5':
                     ['Neck stiffness : my neck movement is restricted due to stiffness', {0:"No", 1:"Yes"}],
            'NeckStiffness0':
                     ['Neck stiffness : No more than I believe is normal', {0:"No", 1:"Yes"}],
            'Headache_YesorNo':
                    ['Headaches', {0:"No", 1:"Yes"}],
             'Headache_Neck':
                    ['Headaches coming from the neck', {0:"No", 1:"Yes"}],
            'Headache_Jaw':
                    ['Headaches coming from the jaw', {0:"No", 1:"Yes"}],
            'EarFullness6':
                    ['Ear fullness : after activity mainly', {0:"No", 1:"Yes"}],
           'EarFullness5':
                    ['Ear fullness : after a bad sleep', {0:"No", 1:"Yes"}],
                 'EarFullness4':
                    ['Ear fullness : after listening to some sounds or being exposed to noise', {0:"No", 1:"Yes"}],
                 'EarFullness3':
                    ['Ear fullness : after working at a computer or desk', {0:"No", 1:"Yes"}],
                 'EarFullness2':
                    ['Ear fullness : after periods of stress / anxiety', {0:"No", 1:"Yes"}],
                 'EarFullness1':
                    ['Ear fullness : yes but no identified cause', {0:"No", 1:"Yes"}],
                 'EarFullness0':
                    ['No ear fullness ', {0:"No", 1:"Yes"}]

            }

JS_str_fields = {'age':['Age (years)', {}],
                 'tschq-months_onset':['Tinnitus duration (months)', {}],
                'tschq-tinpitch':['Tinnitus pitch', {}],
                "tinnitus_laterality":['Tinnitus laterality (0 : bilateral, 0.5 : More on one side, 1 : unilateral)', {}],
                 'tschq-personal_loudness':['Average loudness', {}],
                 'thi_score':['THI score', {}],
                 'tq_score':['TQ score', {}],
                 'ts_1':["How much of a problem is your tinnitus at present? (0 : no problem, 5 : a very big problem", {}],
                 'ts_2':["How STRONG or LOUD is your tinnitus at present? (0 to 10 scale)", {}],
                 'ts_3':["How UNCOMFORTABLE is your tinnitus at present, if everything around you is quiet? (0 to 10 scale)", {}],
                'ts_4':["How ANNOYING is your tinnitus at present? (0 to 10 scale)", {}],
                'ts_5':["How easy is it for you to IGNORE your tinnitus at present? (0 to 10 scale)", {}],
               'ts_6':["How UNPLEASANT is your tinnitus at present? (0 to 10 scale)", {}],
               'tschq-awareness_time':["% of time aware of tinnitus", {}],
                 'tschq-angerness_time':["% of time annoyed, distressed, or irritated of your tinnitus", {}],

                'whoqol_1':["WHOQOL : Physical Health", {}],
                'whoqol_2':["WHOQOL : Psychological Health", {}],
                 'whoqol_3':["WHOQOL : Social Factors", {}],
                 'whoqol_4':["WHOQOL : Environmental Factors", {}],
                'tschq-hyperacusis':["Degree of hyperacusis (0 : None, 4 : Very important)", {}],
                'sex':['Gender', {0:"Male", 1:"Female"}],
                 'tschq-handedness':['Handedness', {0:"Right-handed", 1:"Ambidextrous", 2:"Left-handed"}],
                 'tschq-hearproblems':['Hearing difficulties', {0:"Yes", 1:"No"}],

                 'tschq-tintype':['Tinnitus type of sound', {0:"Tonal", 1:"Noise", 2:"Criquets", 3: "Other"}],
                'tschq-pulsating':['Tinnitus sound pulsatile', {0:"Pulsatile following heart beats",
                                                                1:"Pulsatile but not following heart beats",
                                                                2:"Not pulsatile"}],
                 'tschq-tinside':['Tinnitus side', {0:"Right ear",
                                                    1:"Left ear",
                                                    2 : "Both ears, worse in left",
                                                    3 : "Both ears, worse in right",
                                                    4 : "Both ears, equally",
                                                    5 : "Inside the head",
                                                    6 : "Elsewhere"
                                                    }],
                  'tschq-familiy':['Family history of tinnitus complaints', {0:"Yes", 1:"No"}],
                 'tschq-inital-perception':['Tinnitus onset', {0:"Gradual", 1:"Abrupt"}],
                "tschq-cause_loud_blast":['Tinnitus cause : noise trauma', {0:"Yes", 1:"No"}],
                 "tschq-cause_stress":["Tinnitus cause : psychological (stress, anxiety, depression)", {0:"Yes", 1:"No"}],
                 "tschq-cause_whiplash":['Tinnitus cause : whiplash', {0:"Yes", 1:"No"}],
                 "tschq-cause_head_trauma":['Tinnitus cause : head trauma', {0:"Yes", 1:"No"}],
                "tschq-cause_change_in_hearing":['Tinnitus cause : noise trauma', {0:"Yes", 1:"No"}],
                 "tschq-cause_other":['Tinnitus cause : other', {0:"Yes", 1:"No"}],
                 'tschq-fluctuations':['Tinnitus varies from day to day', {0:"Yes", 1:"No"}],
               'tschq-intermitent':['Tinnitus intermittent or continuous ?', {0:"Intermittent", 1:"Continuous"}],
                 'tschq-stress':['Influence of stress over tinnitus', {0 : "Worsens", 1:"No effect", 2:"Improves"}],
                 'tschq-sleep':['Sleep at night and tinnitus during the day', {0 : "Linked", 1:"Not linked"}],
                 'tschq-sounds-worsen-tinnitus':['Some sounds can worsen tinnitus', {1:"No", 0:"Yes"}],
                 'tschq-sounds-suppress-tinnitus':['Some sounds can suppress tinnitus', {1:"No", 0:"Yes"}],
                 'tschq-noise-pain':['Some sounds can cause physical discomfort', {1:"No", 0:"Yes"}],
                 'tschq-somatic':['Somatosensory : Jaw or head movements can modulate tinnitus', {1:"No", 0:"Yes"}],
               'tschq-tmj':['Temporomandibular disorder', {1:"No", 0:"Yes"}],
                'tschq-neckpain':['Neck pain', {1:"No", 0:"Yes"}],
                 'tschq-headache':['Headaches', {1:"No", 0:"Yes"}],
                'tschq-vertigo':['Vertigo', {1:"No", 0:"Yes"}],
               'tschq-n-treatments':['Number of treatment tested', {0: "0 (none)",
                                                                    1 : "One",
                                                                    2 : "2 to 4",
                                                                    3 : "5 and more",
                                                                    4 : "several",
                                                                    5 : "many"}],
                 'tschq-psychiatry':['Currently under psychiatric treatment', {1:"No", 0:"Yes"}],
                 'tschq-ha':['Hearing aid user', {0 : "right",
                                                    1 : "left",
                                                    2 : "both",
                                                    3 : "none"}]
                 }




