import os
import pandas as pd

SM_num_cols = ['TimeTin-months', 'Age', 'MeanLoudness', 'MinLoudness', 'MaxLoudness', 'CurrLoudness', 'MeanAnnoyancePastWeek',
           'MinAnnoyanceGoodDay', 'MaxAnnoyanceBadDay', 'CurrAnnoyance', 'AwarenessOpenEnded', "Fleeting tinnitus"]
SM_already_categorical=['Female', 'cause_Allergy to something', 'cause_Barotrauma - air / water pressure', 'cause_Dental treatment',
           'cause_Otosclerosis', 'cause_Ear wax procedure (syringing, candling or other related procedure)',
           'cause_Eustachian tube dysfunction', 'cause_Metabolic (diabetes, thyroid, B12, hyperlipidaemia etc.)',
           'cause_Head or neck injury', 'cause_Sudden hearing loss', "cause_Meniere's",
           'cause_Psychological (stress, anxiety, depression)', 'cause_TMJ (issues with the jaw)',
           'cause_Virus (ear infection, flu, cold)', 'cause_Ear wax build up',
           'cause_Ototoxic (from drugs or medication)', 'cause_Noise trauma', 'cause_Noise induced hearing loss',
           'cause_ageRelated', 'cause_Unknown', 'ReactSound_Worse', 'ReactSound_Better', 'HearingLoss',
            'HearingLoss_Echelle','NeckStiffness1', 'NeckStiffness2', 'NeckStiffness3', 'NeckStiffness4',
           'NeckStiffness5', 'NeckStiffness0', 'Headache_YesorNo', 'Headache_Neck', 'Headache_Jaw', 'EarFullness6',
           'EarFullness5', 'EarFullness4', 'EarFullness3', 'EarFullness2', 'EarFullness1', 'EarFullness0', 'Grinding',
           'Clenching', 'TinChange_Pressing the jaw on the side', 'TinChange_Pushing the jaw backwards',
           'TinChange_Pushing the lower jaw outwards rapidly',
           'TinChange_Pushing your hand against your forehead while resisting with the neck muscles',
           'TinChange_Clenching the teeth together', 'TinChange_TiltingÂ\xa0your head backwards',
           'TinChange_No change with any movements', 'JawPain_Yes, my jaw sometimes feels painful',
            'JawPain_Yes, I struggle to fully move my jaw', 'JawPain_Yes, my jaw sometimes feels tired',
           'JawPain_Yes, muscles around my jaw feel tight or tense',
           'JawPain_Yes, I have several popping and clicking noises in my jaw',
           'JawPain_Yes, I have been diagnosed with TMJ dysfunction', 'JawPain_No real issues that I am aware of',
           'TinType_A pure tone / single pitch', 'TinType_A mixture of tones', 'TinType_A low buzzing',
           'TinType_A high buzzing', 'TinType_Electric / interference type sound', 'TinType_A low rumbling',
           'TinType_A static noise', 'TinType_Clicking', 'TinType_Intermittent beeping (morse code)',
           'TinType_A whooshing noise - pulsatile', 'TinType_A whooshing noise - not in time with the pulse',
           'TinType_A pulsating sound', 'TinType_Other',  'InflStress', 'InflAnxiety', 'InflGoodSleep', 'InflBadSleep',
            'InflIntenseworkout', 'InflModerateWorkout', 'InflLightEx']
SM_cols_categ_radio = [['HearingLoss_Severe0', 'HearingLoss_Severe1', 'HearingLoss_Severe3',
           'HearingLoss_Severe4'], ['Fluctuations_On_off', 'Fluctuations_GrowsLouder', 'Fluctuations_GrowsQuieter',
           'Fleeting tinnitus'],
           ['Mask_only by a small selection of sounds', 'Mask_only in the shower or by other loud water type sounds',
           'Mask_by things such as TV, music or general background noise',
           'Mask_by white noise or special masking sounds', 'Mask_by nearly all sounds']]

JS_num_cols = ['age', 'mdi_score', 'tfi_score', 'tfi_intrusive', 'tfi_sense_of_control', 'tfi_cognitive',
                           'tfi_sleep', 'tfi_auditory', 'tfi_relaxation', 'tfi_qol', 'tfi_emotional', 'thi_score',
               'tq_score', 'whoqol_1', 'whoqol_2', 'whoqol_3', 'whoqol_4',  'ts_1', 'ts_2', 'ts_3', 'ts_4', 'ts_5',
               'ts_6', 'tschq-personal_loudness', 'tschq-months_onset',
                           'tschq-awareness_time', 'tschq-angerness_time', ]
JS_categorical_essential = [ 'sex',  'tschq-psychiatry', 'tschq-neckpain',
                           'tschq-tmj', 'tschq-vertigo', 'tschq-headache', 'tschq-noise-pain', 'tschq-hyperacusis',
                           'tschq-ha', 'tschq-hearproblems', 'tschq-stress', 'tschq-sleep',
                           'tschq-somatic', 'tschq-sounds-worsen-tinnitus', 'tschq-sounds-suppress-tinnitus',
                           'tschq-n-treatments', 'tschq-tinpitch', 'tschq-tintype', 'tschq-tinside', 'tschq-fluctuations',
                           'tschq-intermitent', 'tschq-pulsating', 'tschq-cause', 'tschq-innital-perception',
                           'tschq-familiy', "tinnitus_laterality" ]
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