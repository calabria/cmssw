#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "FWCore/Services/src/SiteLocalConfigService.h"

#include "FWCore/Services/src/Tracer.h"
#include "FWCore/Services/interface/cudaService_TBBQueueBlocking.h"
#include "FWCore/Services/src/InitRootHandlers.h"
#include "FWCore/Services/src/UnixSignalService.h"

#include "FWCore/Services/src/JobReportService.h"

#include "FWCore/Services/interface/Timing.h"
#include "FWCore/Services/src/Memory.h"
#include "FWCore/Services/src/CPU.h"
#include "FWCore/Services/src/LoadAllDictionaries.h"
#include "FWCore/Services/src/EnableFloatingPointExceptions.h"
#include "FWCore/Services/interface/PrintLoadingPlugins.h"

using edm::service::JobReportService;
using edm::service::SiteLocalConfigService;

using edm::service::EnableFloatingPointExceptions;
using edm::service::InitRootHandlers;
using edm::service::UnixSignalService;
using edm::service::CudaService;

DEFINE_FWK_SERVICE(Tracer);
DEFINE_FWK_SERVICE(CPU);
DEFINE_FWK_SERVICE(CudaService);
/*typedef edm::serviceregistry::NoArgsMaker<CudaService> CudaServiceMaker;
DEFINE_FWK_SERVICE_MAKER(CudaService, CudaServiceMaker);*/


typedef edm::serviceregistry::ParameterSetMaker<edm::SiteLocalConfig,SiteLocalConfigService> SiteLocalConfigMaker;
DEFINE_FWK_SERVICE_MAKER(SiteLocalConfigService,SiteLocalConfigMaker);
typedef edm::serviceregistry::AllArgsMaker<edm::JobReport,JobReportService> JobReportMaker;
DEFINE_FWK_SERVICE_MAKER(JobReportService, JobReportMaker);
