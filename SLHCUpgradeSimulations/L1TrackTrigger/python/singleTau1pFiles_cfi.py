import FWCore.ParameterSet.Config as cms


singleTau1pFiles = cms.untracked.vstring(
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_1.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_10.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_100.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_101.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_102.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_103.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_104.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_105.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_106.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_107.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_108.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_109.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_11.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_110.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_111.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_112.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_113.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_114.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_115.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_116.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_117.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_118.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_119.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_12.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_121.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_122.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_123.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_124.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_125.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_13.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_14.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_15.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_16.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_17.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_18.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_19.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_2.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_20.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_21.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_22.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_23.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_24.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_25.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_26.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_27.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_28.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_29.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_3.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_30.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_31.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_32.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_33.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_34.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_35.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_36.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_37.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_38.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_39.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_4.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_40.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_41.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_42.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_43.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_44.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_45.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_46.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_47.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_48.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_49.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_5.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_50.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_51.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_52.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_53.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_54.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_55.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_56.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_57.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_58.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_59.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_6.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_60.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_61.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_62.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_63.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_64.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_65.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_66.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_67.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_68.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_69.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_7.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_70.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_71.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_72.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_73.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_74.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_75.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_76.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_77.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_78.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_79.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_8.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_80.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_81.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_82.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_83.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_84.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_85.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_86.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_87.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_88.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_89.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_9.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_90.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_91.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_92.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_93.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_94.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_95.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_96.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_97.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_98.root",
"/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/SingleTau1p/PU140/SingleTau1p_E2023TTI_PU140_99.root"
)
