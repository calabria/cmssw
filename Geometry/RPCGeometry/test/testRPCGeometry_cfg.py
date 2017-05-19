import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("Configuration.Geometry.GeometryExtended2023D12_cff")        # ME0 Geometry with 10 etapartitions
process.load("Configuration.Geometry.GeometryExtended2023D12Reco_cff")    # ME0 Geometry with 10 etapartitions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('FWCore.MessageLogger.MessageLogger_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger = cms.Service("MessageLogger")

process.test1 = cms.EDAnalyzer("RPCGEO")
process.test2 = cms.EDAnalyzer("RPCGeometryAnalyzer")

process.p = cms.Path(process.test2)

