#ifndef ME0OBJECTS_ME0DIGISIMLINK_H
#define ME0OBJECTS_ME0DIGISIMLINK_H

#include <map>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"

class ME0DigiSimLink
{
public:
  ME0DigiSimLink(ME0DigiPreReco digi, Local3DPoint entryPoint, LocalVector momentumAtEntry,
      float timeOfFlight, float energyLoss, int particleType, unsigned int detUnitId, unsigned int trackId,
      EncodedEventId eventId, unsigned short processType)
  {

    _entryPoint = entryPoint;
    _momentumAtEntry = momentumAtEntry;
    _timeOfFlight = timeOfFlight;
    _energyLoss = energyLoss;
    _particleType = particleType;
    _detUnitId = detUnitId;
    _trackId = trackId;
    _eventId = eventId;
    _processType = processType;
    _digi = digi;
  }

  ME0DigiSimLink()    {;}

  ~ME0DigiSimLink()   {;}

  float getX()                      const{return _digi.x();}
  float getY()                      const{return _digi.y();}
  float getErrX()                   const{return _digi.ex();}
  float getErrY()                   const{return _digi.ey();}
  float getCorr()                   const{return _digi.corr();}
  float getTof()                    const{return _digi.tof();}
  int getPdgid()                    const{return _digi.pdgid();}
  int getPrompt()                   const{return _digi.prompt();}

  Local3DPoint getEntryPoint()      const{return _entryPoint;}
  LocalVector getMomentumAtEntry()  const{return _momentumAtEntry;}
  float getTimeOfFlight()           const{return _timeOfFlight;}
  float getEnergyLoss()             const{return _energyLoss;}
  int getParticleType()             const{return _particleType;}
  unsigned int getDetUnitId()       const{return _detUnitId;}
  unsigned int getTrackId()         const{return _trackId;}
  EncodedEventId getEventId()       const{return _eventId;}
  unsigned short getProcessType()   const{return _processType;}

  inline bool operator<(const ME0DigiSimLink& other)    const{return getX() < other.getX();}
    

private:
  ME0DigiPreReco _digi;

  Local3DPoint _entryPoint;
  LocalVector _momentumAtEntry;
  float _timeOfFlight;
  float _energyLoss;
  int _particleType;
  unsigned int _detUnitId;
  unsigned int _trackId;
  EncodedEventId _eventId;
  unsigned short _processType;

};
#endif
