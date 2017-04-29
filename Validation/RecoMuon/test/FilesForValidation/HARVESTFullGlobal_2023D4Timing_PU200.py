# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step4 --conditions auto:phase2_realistic -s HARVESTING:@phase2Validation+@phase2 --era Phase2C2_timing --filein file:'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_inDQM.root',', --scenario pp --filetype DQM --geometry Extended2023D4 --mc -n -1 --fileout file:step4.root',', --python HARVESTFullGlobal_2023D4Timing_PU200.py --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('HARVESTING',eras.Phase2C2_timing)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring(
                            
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_10.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_11.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_12.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_13.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_14.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_15.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_16.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_17.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_18.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_19.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_1.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_20.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_21.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_22.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_23.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_24.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_25.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_26.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_27.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_28.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_29.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_2.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_30.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_31.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_32.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_33.root',
#'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_34.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_35.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_36.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_37.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_38.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_39.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_3.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_40.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_41.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_42.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_43.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_44.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_45.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_46.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_47.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_48.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_49.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_4.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_50.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_51.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_52.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_53.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_54.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_55.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_56.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_57.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_58.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_59.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_5.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_60.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_61.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_62.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_63.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_64.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_65.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_66.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_67.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_68.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_69.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_6.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_70.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_71.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_72.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_73.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_74.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_75.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_76.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_77.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_78.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_79.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_7.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_80.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_81.root',
'/store/user/calabria/RelValZMM_14/crab_Val_900_pre4_PU200_D4_7/170426_093330/0000/step31_8.root',
                            
                            )
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step4 nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.dqmSaver.workflow = "/Global/CMSSW_9_0_0_pre4/RECO_PU200_D4Timing_ZMM"

# Path and EndPath definitions
process.dqmHarvestingPOGMC = cms.Path(process.DQMOffline_SecondStep_PrePOGMC)
process.validationHarvesting = cms.Path(process.postValidation+process.hltpostvalidation+process.postValidation_gen)
process.validationprodHarvesting = cms.Path(process.hltpostvalidation_prod+process.postValidation_gen)
process.dqmHarvestingFakeHLT = cms.Path(process.DQMOffline_SecondStep_FakeHLT+process.DQMOffline_Certification)
process.validationHarvestingMiniAOD = cms.Path(process.JetPostProcessor+process.METPostProcessorHarvesting+process.postValidationMiniAOD)
process.validationHarvestingHI = cms.Path(process.postValidationHI)
process.genHarvesting = cms.Path(process.postValidation_gen)
process.dqmHarvestingPOG = cms.Path(process.DQMOffline_SecondStep_PrePOG)
process.alcaHarvesting = cms.Path()
process.dqmHarvesting = cms.Path(process.DQMOffline_SecondStep+process.DQMOffline_Certification)
process.validationHarvestingFS = cms.Path(process.postValidation+process.hltpostvalidation+process.postValidation_gen)
process.validationpreprodHarvesting = cms.Path(process.postValidation_preprod+process.hltpostvalidation_preprod+process.postValidation_gen)
process.postValidation_common_step = cms.Path(process.postValidation_common)
process.postValidation_muons_step = cms.Path(process.postValidation_muons)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(process.postValidation_common_step,process.postValidation_muons_step,process.dqmsave_step)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
