import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")
#process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.load("Validation.Configuration.postValidation_cff")
process.load("Validation.RecoMuon.PostProcessorHLT_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(       
	'file:validationEDM_case1.root'
	#'file:root://xrootd.unl.edu//store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_Case2/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_VAL2_Case2/defc9ce5c12ca10cbd1533a5d2d2e5bd/validationEDM_1_1_nfR.root',
       	#'file:root://xrootd.unl.edu//store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_Case2/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC12_2023Scenario_VAL2_Case2/defc9ce5c12ca10cbd1533a5d2d2e5bd/validationEDM_2_1_Xl7.root'
	)
)

process.DQMStore.referenceFileName = ""
process.DQMStore.collateHistograms = False

process.dqmSaver.convention = "Offline"
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings
process.dqmSaver.workflow = "/GlobalValidation/Test/RECO"

from Validation.RecoMuon.PostProcessor_cff import *
process.p1 = cms.Path(process.EDMtoMEConverter*
                      #process.postValidation*
		      process.recoMuonPostProcessors*
#                     process.recoMuonPostProcessorsHLT*
                      process.dqmSaver)
