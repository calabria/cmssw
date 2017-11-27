import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('MyDigis',eras.Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('CommonTools.ParticleFlow.EITopPAG_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                                              'file:step2.root',
#                                                              '/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/GEN-SIM-RECO/PU25ns_94X_mc2017_realistic_v4_highPU_AVE50-v1/10000/22E2A744-E3BA-E711-A1A8-5065F3815241.root',
                                                              ),
    secondaryFileNames = cms.untracked.vstring(
#                                               '/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_94X_mc2017_realistic_v4_highPU_AVE50-v1/10000/62236337-BEBA-E711-9962-4C79BA1810EB.root',
#                                               '/store/relval/CMSSW_9_4_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_94X_mc2017_realistic_v4_highPU_AVE50-v1/10000/D6384D37-BEBA-E711-B24C-4C79BA180B9F.root',
                                               )
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:digi.root'),
#    outputCommands = cms.untracked.vstring("drop *", "keep *_simSiPixelDigis_*_*", "keep *_siPixelDigisGPU_*_*"),
    outputCommands = cms.untracked.vstring("keep *"),
    splitLevel = cms.untracked.int32(0)
)


process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_design', '')

process.siPixelClustersPreSplitting.src = cms.InputTag("siPixelDigisGPU")

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.siPixelDigis)
process.raw2digiGPU_step = cms.Path(process.siPixelDigisGPU)
process.clustering = cms.Path(process.siPixelClustersPreSplitting)
process.RECOSIMoutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)


# Schedule definition
process.schedule = cms.Schedule(#process.raw2digi_step,
                                process.raw2digiGPU_step,
#                                process.clustering,
                                process.RECOSIMoutput_step)


# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions
#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC

#call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllMC(process)

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion


