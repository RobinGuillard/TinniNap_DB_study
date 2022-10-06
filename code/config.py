import os
import pandas as pd

SM_num_cols = ['TimeTin-months', 'Age', 'MeanLoudness', 'MinLoudness', 'MaxLoudness', 'CurrLoudness', 'MeanAnnoyancePastWeek',
           'MinAnnoyanceGoodDay', 'MaxAnnoyanceBadDay', 'CurrAnnoyance', 'AwarenessOpenEnded']
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
