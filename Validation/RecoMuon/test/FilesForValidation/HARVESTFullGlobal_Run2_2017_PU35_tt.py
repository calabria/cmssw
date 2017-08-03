# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step4 --filetype DQM --conditions auto:run2_mc --mc -s HARVESTING:@baseValidation+@muonOnlyValidation --era Run2_2016 -n 100 --filein file:step3_inDQM.root --fileout file:step4.root --python HARVESTFullGlobal_run2_PU0.py --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('HARVESTING',eras.Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring(
                            
    'file:/lustre/cms/store/user/calabria/RelValTTbar_13/crab_Val_911_RUN2_2017_PU_tt_8/170719_085206/0000/step31_1.root',
    'file:/lustre/cms/store/user/calabria/RelValTTbar_13/crab_Val_911_RUN2_2017_PU_tt_8/170719_085206/0000/step31_2.root',
    'file:/lustre/cms/store/user/calabria/RelValTTbar_13/crab_Val_911_RUN2_2017_PU_tt_8/170719_085206/0000/step31_3.root',
    'file:/lustre/cms/store/user/calabria/RelValTTbar_13/crab_Val_911_RUN2_2017_PU_tt_8/170719_085206/0000/step31_4.root',
    'file:/lustre/cms/store/user/calabria/RelValTTbar_13/crab_Val_911_RUN2_2017_PU_tt_8/170719_085206/0000/step31_5.root',
    'file:/lustre/cms/store/user/calabria/RelValTTbar_13/crab_Val_911_RUN2_2017_PU_tt_8/170719_085206/0000/step31_6.root',
    'file:/lustre/cms/store/user/calabria/RelValTTbar_13/crab_Val_911_RUN2_2017_PU_tt_8/170719_085206/0000/step31_7.root',
    'file:/lustre/cms/store/user/calabria/RelValTTbar_13/crab_Val_911_RUN2_2017_PU_tt_8/170719_085206/0000/step31_8.root',
    'file:/lustre/cms/store/user/calabria/RelValTTbar_13/crab_Val_911_RUN2_2017_PU_tt_8/170719_085206/0000/step31_9.root',
#    'file:/lustre/cms/store/user/calabria/RelValTTbar_13/crab_Val_911_RUN2_2017_PU_tt_8/170719_085206/0000/step31_10.root',
                            
                            )
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step4 nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

process.dqmSaver.workflow = "/Global/CMSSW_9_1_1/RECO_PU35_2017_tt"

# Path and EndPath definitions
process.validationHarvestingHI = cms.Path(process.postValidationHI)
process.alcaHarvesting = cms.Path()
process.validationHarvestingFS = cms.Path(process.postValidation+process.hltpostvalidation+process.postValidation_gen)
process.validationpreprodHarvesting = cms.Path(process.postValidation_preprod+process.hltpostvalidation_preprod+process.postValidation_gen)
process.genHarvesting = cms.Path(process.postValidation_gen)
process.dqmHarvestingPOG = cms.Path(process.DQMOffline_SecondStep_PrePOG)
process.validationprodHarvesting = cms.Path(process.hltpostvalidation_prod+process.postValidation_gen)
process.validationHarvesting = cms.Path(process.postValidation+process.hltpostvalidation+process.postValidation_gen)
process.dqmHarvestingPOGMC = cms.Path(process.DQMOffline_SecondStep_PrePOGMC)
process.dqmHarvestingFakeHLT = cms.Path(process.DQMOffline_SecondStep_FakeHLT+process.DQMOffline_Certification)
process.validationHarvestingMiniAOD = cms.Path(process.JetPostProcessor+process.METPostProcessorHarvesting+process.postValidationMiniAOD)
process.dqmHarvesting = cms.Path(process.DQMOffline_SecondStep+process.DQMOffline_Certification)
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
