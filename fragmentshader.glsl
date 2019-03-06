#version 330 core

// Interpolated values from the vertex shaders
in vec3 fragmentColor;
in vec2 fragmentCoordinate;

// Ouput data
out vec3 color;

// Values that stay constant for the whole mesh.
uniform sampler3D g_tex3DParticleDensityLUT;
uniform sampler3D g_tex3DMultipleScatteringInParticleLUT;

/*
 *  GLOBAL VARIABLES: 
 *  ================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
#define PI                              3.1415928f
#define FLT_MAX                   3.402823466e+38f
#define EARTH_RADIUS                     6360000.f
#define ATM_TOP_HEIGHT                     80000.f
#define ATM_TOP_RADIUS                     (EARTH_RADIUS+ATM_TOP_HEIGHT)
#define OPTICAL_DEPTH_LUT_DIM              vec4(64,32,64,32)
#define PARTICLE_SCALE_HEIGHT              vec2(7994.f,1200.f)
#define SRF_SCATTERING_IN_PARTICLE_LUT_DIM vec3(32,64,16)
#define VOL_SCATTERING_IN_PARTICLE_LUT_DIM vec4(32,64,32,8)

#define LERP(v0,v1,t) (1-t)*v0 + t*v1
#define SAMPLE_4D_LUT(tex3D, LUT_DIM, f4LUTCoords, Result){   \
    vec3  f3UVW;                                                    \
    f3UVW.xy = f4LUTCoords.xy;                                      \
    float fQSlice = f4LUTCoords.w * LUT_DIM.w - 0.5;                \
    float fQ0Slice = floor(fQSlice);                                \
    float fQWeight = fQSlice - fQ0Slice;                            \
                                                                    \
    f3UVW.z = (fQ0Slice + f4LUTCoords.z) / LUT_DIM.w;               \
                                                                    \
    Result = LERP( texture3D(tex3D,      f3UVW                          ).r,  \ 
                   texture3D(tex3D, frac(f3UVW + vec3(0,0,1/LUT_DIM.w)) ).r,  \ 
                   fQWeight);                                                 \
}

float _fCloudAltitude      = 3000.f;
float _fCloudThickness     =  700.f;
float _fAttenuationCoeff   =  0.07f; // Typical scattering coefficient lies in the range 0.01 - 0.1 m^-1
float _fParticleCutOffDist =  2e+5f;
float _fEarthRadius     = 6360000.f;

// Fraction of the particle cut off distance which serves as
// a transition region from particles to flat clouds
static const float g_fParticleToFlatMorphRatio = 0.2;

/*
 *  STRUCTURES: 
 *  ==========  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
struct SCloudCellAttribs{
    vec3  f3Center;
    float fSize;

    vec3 f3Normal;
    uint uiNumActiveLayers;

    vec3  f3Tangent;
    float fDensity;

    vec3  f3Bitangent;
    float fMorphFadeout;

    uint uiPackedLocation;
};

struct SParticleAttribs{
    vec3  f3Pos;
    float fSize;
    float fRndAzimuthBias;
    float fDensity;
};

struct SCloudParticleLighting{
    vec4 f4SunLight;
	vec2 f2SunLightAttenuation; // x ==   Direct Sun Light Attenuation
							    // y == Indirect Sun Light Attenuation
    vec4 f4AmbientLight;
};


/*
 *  GENERAL: 
 *  =======  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */



// Computes the zenith and azimuth angles in XZY (Y Up) coordinate system from direction
void DirectionToZenithAzimuthAngleXZY(in vec3 f3Direction, out float fZenithAngle, out float fAzimuthAngle){
    float fZenithCos = f3Direction.y;
    fZenithAngle = acos(fZenithCos);
    //float fZenithSin = sqrt( max(1 - fZenithCos*fZenithCos, 1e-10) );
    float fAzimuthCos = f3Direction.x;// / fZenithSin;
    float fAzimuthSin = f3Direction.z;// / fZenithSin;
    fAzimuthAngle = atan2(fAzimuthSin, fAzimuthCos);
}

// Constructs local XYZ (Z Up) frame from Up and Inward vectors
void ConstructLocalFrameXYZ(in vec3 f3Up, in vec3 f3Inward, out vec3 f3X, out vec3 f3Y, out vec3 f3Z){
    //      Z (Up)
    //      |    Y  (Inward)
    //      |   /
    //      |  /
    //      | /  
    //      |/
    //       -----------> X
    //
    f3Z = normalize(f3Up);
    f3X = normalize(cross(f3Inward, f3Z));
    f3Y = normalize(cross(f3Z, f3X));
}

// Computes zenith and azimuth angles in local XYZ (Z Up) frame from the direction
void ComputeLocalFrameAnglesXYZ(in  vec3  f3LocalX, 
                                in  vec3  f3LocalY, 
                                in  vec3  f3LocalZ,
                                in  vec3  f3RayDir,
                                out float fLocalZenithAngle,
                                out float fLocalAzimuthAngle){
    fLocalZenithAngle = acos(saturate( dot(f3LocalZ, f3RayDir) ));

    // Compute azimuth angle in the local frame
    float fViewDirLocalAzimuthCos = dot(f3RayDir, f3LocalX);
    float fViewDirLocalAzimuthSin = dot(f3RayDir, f3LocalY);
    fLocalAzimuthAngle = atan2(fViewDirLocalAzimuthSin, fViewDirLocalAzimuthCos);
}



// GetRaySphereIntersection
// ........................
// http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
#define NO_INTERSECTIONS vec2(-1,-2)

void GetRaySphereIntersection(in  vec3  f3RayOrigin,
                              in  vec3  f3RayDirection,
                              in  vec3  f3SphereCenter,
                              in  float fSphereRadius,
                              out vec2  f2Intersections){
    f3RayOrigin -= f3SphereCenter;
    float A = dot(f3RayDirection, f3RayDirection);
    float B = 2 * dot(f3RayOrigin, f3RayDirection);
    float C = dot(f3RayOrigin,f3RayOrigin) - fSphereRadius*fSphereRadius;
    float D = B*B - 4*A*C;
    // If discriminant is negative, there are no real roots hence the ray misses the
    // sphere
    if( D<0 ){
        f2Intersections = NO_INTERSECTIONS;
    }
    else{
        D = sqrt(D);
        f2Intersections = vec2(-B - D, -B + D) / (2*A); // A must be positive here!!
    }
}
void GetRaySphereIntersection2(in vec3 f3RayOrigin,
                               in vec3 f3RayDirection,
                               in vec3 f3SphereCenter,
                               in vec2 f2SphereRadius,
                               out vec4 f4Intersections){
    f3RayOrigin -= f3SphereCenter;
    float A = dot(f3RayDirection, f3RayDirection);
    float B = 2 * dot(f3RayOrigin, f3RayDirection);
    vec2  C = dot(f3RayOrigin,f3RayOrigin) - f2SphereRadius*f2SphereRadius;
    vec2  D = B*B - 4*A*C;
    // If discriminant is negative, there are no real roots hence the ray misses the
    // sphere
    vec2 f2RealRootMask = (D.xy >= 0);
    D = sqrt( max(D,0) );
    f4Intersections =   f2RealRootMask.xxyy * vec4(-B - D.x, -B + D.x, -B - D.y, -B + D.y) / (2*A) + 
                      (1-f2RealRootMask.xxyy) * NO_INTERSECTIONS.xyxy;
}



// This function for analytical evaluation of particle density integral is 
// provided by Eric Bruneton
// http://www-evasion.inrialpes.fr/Membres/Eric.Bruneton/
//
// optical depth for ray (r,mu) of length d, using analytic formula
// (mu=cos(view zenith angle)), intersections with ground ignored
vec2 GetDensityIntegralAnalytic(float r, float mu, float d) {
    vec2 f2A = sqrt( (0.5/PARTICLE_SCALE_HEIGHT.xy) * r );
    vec4 f4A01 = f2A.xxyy * vec2(mu, mu + d / r).xyxy;
    vec4 f4A01s = sign(f4A01);
    vec4 f4A01sq = f4A01*f4A01;
    
    vec2 f2X;
    f2X.x = f4A01s.y > f4A01s.x ? exp(f4A01sq.x) : 0.0;
    f2X.y = f4A01s.w > f4A01s.z ? exp(f4A01sq.z) : 0.0;
    
    vec4 f4Y = f4A01s / (2.3193*abs(f4A01) + sqrt(1.52*f4A01sq + 4.0)) * vec3(1.0, exp(-d/PARTICLE_SCALE_HEIGHT.xy*(d/(2.0*r)+mu))).xyxz;

    return sqrt((6.2831*PARTICLE_SCALE_HEIGHT)*r) * exp((EARTH_RADIUS-r)/PARTICLE_SCALE_HEIGHT.xy) * (f2X + vec2( dot(f4Y.xy, vec2(1.0, -1.0)), dot(f4Y.zw, vec2(1.0, -1.0)) ));
}

vec3 GetExtinctionUnverified(in vec3 f3StartPos, in vec3 f3EndPos, vec3 f3EyeDir, vec3 f3EarthCentre){

    float r = length(f3StartPos-f3EarthCentre);
    float fCosZenithAngle = dot(f3StartPos-f3EarthCentre, f3EyeDir) / r;
    vec2  f2ParticleDensity = GetDensityIntegralAnalytic(r, fCosZenithAngle, length(f3StartPos - f3EndPos));

    // Get optical depth
    vec3 f3TotalRlghOpticalDepth = _f4RayleighExtinctionCoeff.xyz * f2ParticleDensity.x; // ----------------------------------------------------------------------------------- _f4RayleighExtinctionCoeff
    vec3 f3TotalMieOpticalDepth  = _f4MieExtinctionCoeff     .xyz * f2ParticleDensity.y; // ----------------------------------------------------------------------------------- _f4MieExtinctionCoeff
        
    // Compute extinction
    vec3   f3Extinction = exp( -(f3TotalRlghOpticalDepth + f3TotalMieOpticalDepth) );
    return f3Extinction;
}

vec3 GetExtinction(in vec3 f3StartPos, in vec3 f3EyeDir, in float fRayLength){
    vec3 f3EarthCentre = /*g_CameraAttribs.f4CameraPos.xyz*vec3(1,0,1)*/ - vec3(0,1,0) * EARTH_RADIUS;

    vec2 f2RayAtmTopIsecs; 
    // Compute intersections of the view ray with the atmosphere
    GetRaySphereIntersection(f3StartPos, f3EyeDir, f3EarthCentre, ATM_TOP_RADIUS, f2RayAtmTopIsecs);
    // If the ray misses the atmosphere, there is no extinction
    if( f2RayAtmTopIsecs.y < 0 )return 1;

    // Do not let the start and end point be outside the atmosphere
    vec3 f3EndPos = f3StartPos + f3EyeDir * min(f2RayAtmTopIsecs.y, fRayLength);
    f3StartPos += f3EyeDir * max(f2RayAtmTopIsecs.x, 0);

    return GetExtinctionUnverified(f3StartPos, f3EndPos, f3EyeDir, f3EarthCentre);
}

vec3 GetExtinction(in vec3 f3StartPos, in vec3 f3EndPos){
    vec3  f3EyeDir = f3EndPos - f3StartPos;
    float fRayLength = length(f3EyeDir);
    f3EyeDir /= fRayLength;

    return GetExtinction(f3StartPos, f3EyeDir, fRayLength);
}

float GetCosHorizonAnlge(float fHeight){
    // Due to numeric precision issues, fHeight might sometimes be slightly negative
    fHeight = max(fHeight, 0);
    return -sqrt(fHeight * (2*EARTH_RADIUS + fHeight) ) / (EARTH_RADIUS + fHeight);
}

float ZenithAngle2TexCoord(float fCosZenithAngle, float fHeight, in float fTexDim, float power, float fPrevTexCoord){
    fCosZenithAngle = fCosZenithAngle;
    float fTexCoord;
    float fCosHorzAngle = GetCosHorizonAnlge(fHeight);
    // When performing look-ups into the scattering texture, it is very important that all the look-ups are consistent
    // wrt to the horizon. This means that if the first look-up is above (below) horizon, then the second look-up
    // should also be above (below) horizon. 
    // We use previous texture coordinate, if it is provided, to find out if previous look-up was above or below
    // horizon. If texture coordinate is negative, then this is the first look-up
    bool bIsAboveHorizon = fPrevTexCoord >= 0.5;
    bool bIsBelowHorizon = 0 <= fPrevTexCoord && fPrevTexCoord < 0.5;
    if(  bIsAboveHorizon || !bIsBelowHorizon && (fCosZenithAngle > fCosHorzAngle) ){
        // Scale to [0,1]
        fTexCoord = saturate( (fCosZenithAngle - fCosHorzAngle) / (1 - fCosHorzAngle) );
        fTexCoord = pow(fTexCoord, power);
        // Now remap texture coordinate to the upper half of the texture.
        // To avoid filtering across discontinuity at 0.5, we must map
        // the texture coordinate to [0.5 + 0.5/fTexDim, 1 - 0.5/fTexDim]
        //
        //      0.5   1.5               D/2+0.5        D-0.5  texture coordinate x dimension
        //       |     |                   |            |
        //    |  X  |  X  | .... |  X  ||  X  | .... |  X  |  
        //       0     1          D/2-1   D/2          D-1    texel index
        //
        fTexCoord = 0.5f + 0.5f / fTexDim + fTexCoord * (fTexDim/2 - 1) / fTexDim;
    }
    else{
        fTexCoord = saturate( (fCosHorzAngle - fCosZenithAngle) / (fCosHorzAngle - (-1)) );
        fTexCoord = pow(fTexCoord, power);
        // Now remap texture coordinate to the lower half of the texture.
        // To avoid filtering across discontinuity at 0.5, we must map
        // the texture coordinate to [0.5, 0.5 - 0.5/fTexDim]
        //
        //      0.5   1.5        D/2-0.5             texture coordinate x dimension
        //       |     |            |       
        //    |  X  |  X  | .... |  X  ||  X  | .... 
        //       0     1          D/2-1   D/2        texel index
        //
        fTexCoord = 0.5f / fTexDim + fTexCoord * (fTexDim/2 - 1) / fTexDim;
    }    

    return fTexCoord;
}

float HGPhaseFunc(in float fCosTheta, in const float g = 0.9){
    return (1/(4*PI) * (1 - g*g)) / pow( max((1 + g*g) - (2*g)*fCosTheta,0), 3.f/2.f);
}









void WorldParamsToOpticalDepthLUTCoords(in vec3 f3NormalizedStartPos, in vec3 f3RayDir, out vec4 f4LUTCoords){
    
    DirectionToZenithAzimuthAngleXZY(f3NormalizedStartPos, f4LUTCoords.x, f4LUTCoords.y);

    vec3 f3LocalX, f3LocalY, f3LocalZ;
    // Construct local tangent frame for the start point on the sphere (z up)
    // For convinience make the Z axis look into the sphere
    ConstructLocalFrameXYZ( -f3NormalizedStartPos, vec3(0,1,0), f3LocalX, f3LocalY, f3LocalZ);

    // z coordinate is the angle between the ray direction and the local frame zenith direction
    // Note that since we are interested in rays going inside the sphere only, the allowable
    // range is [0, PI/2]

    float fRayDirLocalZenith, fRayDirLocalAzimuth;
    ComputeLocalFrameAnglesXYZ(f3LocalX, f3LocalY, f3LocalZ, f3RayDir, fRayDirLocalZenith, fRayDirLocalAzimuth);
    f4LUTCoords.z = fRayDirLocalZenith;
    f4LUTCoords.w = fRayDirLocalAzimuth;

    f4LUTCoords.xyzw = f4LUTCoords.xyzw / vec4(PI, 2*PI, PI/2, 2*PI) + vec4(0.0, 0.5, 0, 0.5);

    // Clamp only zenith (yz) coordinate as azimuth is filtered with wraparound mode
    f4LUTCoords.xz = clamp(f4LUTCoords, 0.5/OPTICAL_DEPTH_LUT_DIM, 1.0-0.5/OPTICAL_DEPTH_LUT_DIM).xz;

}


// All parameters must be defined in the unit sphere (US) space
vec4 WorldParamsToParticleScatteringLUT( in vec3 f3StartPosUSSpace, 
                                         in vec3 f3ViewDirInUSSpace, 
                                         in vec3 f3LightDirInUSSpace){
    vec4 f4LUTCoords = 0;
    bool bSurfaceOnly = true;

    float fDistFromCenter = 0;
    if( !bSurfaceOnly ){
        // Compute distance from center and normalize start position
        fDistFromCenter = length(f3StartPosUSSpace);
        f3StartPosUSSpace /= max(fDistFromCenter, 1e-5);
    }
    float fStartPosZenithCos = dot(f3StartPosUSSpace, f3LightDirInUSSpace);
    f4LUTCoords.x = acos(fStartPosZenithCos);

    vec3 f3LocalX, f3LocalY, f3LocalZ;
    ConstructLocalFrameXYZ(-f3StartPosUSSpace, f3LightDirInUSSpace, f3LocalX, f3LocalY, f3LocalZ);

    float fViewDirLocalZenith, fViewDirLocalAzimuth;
    ComputeLocalFrameAnglesXYZ(f3LocalX, f3LocalY, f3LocalZ, f3ViewDirInUSSpace, fViewDirLocalZenith, fViewDirLocalAzimuth);
    f4LUTCoords.y = fViewDirLocalAzimuth;
    f4LUTCoords.z = fViewDirLocalZenith;
    
    // In case the parameterization is performed for the sphere surface, the allowable range for the 
    // view direction zenith angle is [0, PI/2] since the ray should always be directed into the sphere.
    // Otherwise the range is whole [0, PI]
    f4LUTCoords.xyz = f4LUTCoords.xyz / vec3(PI, 2*PI, bSurfaceOnly ? (PI/2) : PI) + vec3(0, 0.5, 0);
    if( bSurfaceOnly )
        f4LUTCoords.w = 0;
    else
        f4LUTCoords.w = fDistFromCenter;
    if( bSurfaceOnly )
        f4LUTCoords.xz = clamp(f4LUTCoords.xyz, 0.5/SRF_SCATTERING_IN_PARTICLE_LUT_DIM, 1-0.5/SRF_SCATTERING_IN_PARTICLE_LUT_DIM).xz;
    else
        f4LUTCoords.xzw = clamp(f4LUTCoords, 0.5/VOL_SCATTERING_IN_PARTICLE_LUT_DIM, 1-0.5/VOL_SCATTERING_IN_PARTICLE_LUT_DIM).xzw;

    return f4LUTCoords;
}



/*
 *  SINGLE SCATTERING: 
 *  =================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
 
// This shader pre-computes the radiance of single scattering at a given point in given
// direction.
/*
vec3 PrecomputeSingleScatteringPS(){
    // Get attributes for the current point
    vec2 f2UV = ProjToUV(In.m_f2PosPS);
    float fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle;
    InsctrLUTCoords2WorldParams(vec4(f2UV, g_MiscParams.f2WQ), fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle );
    vec3 f3EarthCentre =  - vec3(0,1,0) * EARTH_RADIUS;
    vec3 f3RayStart    = vec3(0, fHeight, 0);
    vec3 f3ViewDir     = ComputeViewDir (fCosViewZenithAngle);
    vec3 f3DirOnLight  = ComputeLightDir(f3ViewDir, fCosSunZenithAngle, fCosSunViewAngle);
  
    // Intersect view ray with the top of the atmosphere and the Earth
    vec4 f4Isecs;
    GetRaySphereIntersection2( f3RayStart, f3ViewDir, f3EarthCentre, 
                               vec2(EARTH_RADIUS, ATM_TOP_RADIUS), 
                               f4Isecs);
    vec2 f2RayEarthIsecs  = f4Isecs.xy;
    vec2 f2RayAtmTopIsecs = f4Isecs.zw;

    if(f2RayAtmTopIsecs.y <= 0)
        return 0; // This is just a sanity check and should never happen
                  // as the start point is always under the top of the 
                  // atmosphere (look at InsctrLUTCoords2WorldParams())

    // Set the ray length to the distance to the top of the atmosphere
    float fRayLength = f2RayAtmTopIsecs.y;
    // If ray hits Earth, limit the length by the distance to the surface
    if(f2RayEarthIsecs.x > 0)
        fRayLength = min(fRayLength, f2RayEarthIsecs.x);
    
    vec3 f3RayEnd = f3RayStart + f3ViewDir * fRayLength;

    float fCloudTransparency =  1      ;
    float fDitToCloud        = +FLT_MAX;
    // Integrate single-scattering
    vec3 f3Inscattering, f3Extinction;
    IntegrateUnshadowedInscattering(f3RayStart, 
                                    f3RayEnd,
                                    f3ViewDir,
                                    f3EarthCentre,
                                    f3DirOnLight.xyz,
                                    100,
                                    f3Inscattering,
                                    f3Extinction,
                                    fCloudTransparency,
                                    fDitToCloud);

    return f3Inscattering;
}
*/

/*
 * COMPUTE PARTICLES:
 * =================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This function computes different attributes of a particle which will be used
 * for rendering
 *
 */
void ComputeParticleRenderAttribs(  const in SParticleAttribs       ParticleAttrs,
                                    const in SCloudCellAttribs      CellAttrs,
                                          in SCloudParticleLighting ParticleLighting, 
                                    in vec3  f3CameraPos,
                                    in vec3  f3ViewRay,
                                    in vec3  f3EntryPointUSSpace, // Ray entry point in unit sphere (US) space
                                    in vec3  f3ViewRayUSSpace,    // View direction in unit sphere (US) space
                                    in vec3  f3LightDirUSSpace,   // Light direction in unit sphere (US) space
                                    in float fDistanceToExitPoint,
                                    in float fDistanceToEntryPoint,
                                    out vec4 f4Color
                                ){
    vec3 f3EntryPointWS = f3CameraPos + fDistanceToEntryPoint * f3ViewRay;
    vec3 f3ExitPointWS  = f3CameraPos +  fDistanceToExitPoint * f3ViewRay;

	// Compute look-up coordinates
    vec4 f4LUTCoords;
    WorldParamsToOpticalDepthLUTCoords(f3EntryPointUSSpace, f3ViewRayUSSpace, f4LUTCoords);

    // Randomly rotate the sphere
    f4LUTCoords.y += ParticleAttrs.fRndAzimuthBias;

	// Get the normalized density along the view ray
    float fNormalizedDensity = 1.f;

    SAMPLE_4D_LUT(g_tex3DParticleDensityLUT, OPTICAL_DEPTH_LUT_DIM, f4LUTCoords, fNormalizedDensity);

	// Compute actual cloud mass by multiplying normalized density with ray length
    fCloudMass = fNormalizedDensity * (fDistanceToExitPoint - fDistanceToEntryPoint);
    float fFadeOutDistance = _fParticleCutOffDist * g_fParticleToFlatMorphRatio;
    float fFadeOutFactor = saturate( (_fParticleCutOffDist - fDistanceToEntryPoint) /  max(fFadeOutDistance,1) );
    fCloudMass *= fFadeOutFactor * CellAttrs.fMorphFadeout;
    fCloudMass *= ParticleAttrs.fDensity;

	// Compute transparency
    fTransparency = exp( -fCloudMass * _fAttenuationCoeff );
    
	// Evaluate phase function for single scattering
	float fCosTheta = dot(-f3ViewRayUSSpace, f3LightDirUSSpace);
	float PhaseFunc = HGPhaseFunc(fCosTheta, 0.8);

	vec2 f2SunLightAttenuation = ParticleLighting.f2SunLightAttenuation;
	vec3 f3SingleScattering =  fTransparency *  ParticleLighting.f4SunLight.xyz * f2SunLightAttenuation.x * PhaseFunc * pow(CellAttrs.fMorphFadeout,2);

    // Multiple Scattering
	vec4  f4MultipleScatteringLUTCoords = WorldParamsToParticleScatteringLUT(f3EntryPointUSSpace, f3ViewRayUSSpace, f3LightDirUSSpace);

    float fMultipleScattering = texture3D(g_tex3DMultipleScatteringInParticleLUT,  f4MultipleScatteringLUTCoords.xyz).r; 
	vec3  f3MultipleScattering = (1-fTransparency) * fMultipleScattering * f2SunLightAttenuation.y * ParticleLighting.f4SunLight.xyz;

	// Compute ambient light
	vec3  f3EarthCentre = vec3(0, -_fEarthRadius, 0);
	float fEnttryPointAltitude = length(f3EntryPointWS - f3EarthCentre);
	float fCloudBottomBoundary = _fEarthRadius + _fCloudAltitude - _fCloudThickness/2.f;
	float fAmbientStrength     =  (fEnttryPointAltitude - fCloudBottomBoundary) /  _fCloudThickness;//(1-fNoise)*0.5;//0.3;
	fAmbientStrength = clamp(fAmbientStrength, 0.3, 1.0);
	vec3 f3Ambient = (1-fTransparency) * fAmbientStrength * ParticleLighting.f4AmbientLight.xyz;

	// Compose color
	f4Color.xyz = 0;
	const float fSingleScatteringScale = 0.2;
	f4Color.xyz += f3SingleScattering * fSingleScatteringScale;
	f4Color.xyz += f3MultipleScattering * PI;
	f4Color.xyz += f3Ambient;
	f4Color.xyz *= 2;

    f4Color.w = fTransparency;
}



// This shader renders particles
void RenderCloudsPS(PS_Input In,
                    out float fTransparency : SV_Target0,
                    out float fDistToCloud  : SV_Target1,
                    out vec4  f4Color       : SV_Target2
                    ){

    SParticleAttribs          ParticleAttrs = g_Particles          [In.uiParticleID];
    SCloudCellAttribs           CellAttribs = g_CloudCells         [In.uiParticleID / g_GlobalCloudAttribs.uiMaxLayers];
    SCloudParticleLighting ParticleLighting = g_bufParticleLighting[In.uiParticleID];

    vec3 f3CameraPos, f3ViewRay;

    f3CameraPos = g_CameraAttribs.f4CameraPos.xyz;  // Posicion de camara
    f3ViewRay   =         normalize(In.f3ViewRay);  // direccion de vista?
    
    vec2 f2ScreenDim = vec2(1024, 768);
    vec2 f2PosPS = UVToProj( In.f4Pos.xy / f2ScreenDim );
    
    float fDepth = GetConservativeScreenDepth( ProjToUV(f2PosPS.xy) );
    vec4  f4ReconstructedPosWS = mul( ve4(f2PosPS.xy,fDepth,1.0), g_CameraAttribs.mViewProjInv );
    vec3  f3WorldPos = f4ReconstructedPosWS.xyz / f4ReconstructedPosWS.w;

    // Compute view ray
    f3ViewRay = f3WorldPos - f3CameraPos;
    float fRayLength = length(f3ViewRay);
    f3ViewRay /= fRayLength;

    // Intersect view ray with the particle
    vec2  f2RayIsecs;
    float fDistanceToEntryPoint, fDistanceToExitPoint;
    vec3  f3EntryPointUSSpace, f3ViewRayUSSpace, f3LightDirUSSpace;
    IntersectRayWithParticle(ParticleAttrs, CellAttribs, f3CameraPos,  f3ViewRay,
                             f2RayIsecs, f3EntryPointUSSpace, f3ViewRayUSSpace,
                             f3LightDirUSSpace,
                             fDistanceToEntryPoint, fDistanceToExitPoint);
   
    if( f2RayIsecs.y < 0 || fRayLength < fDistanceToEntryPoint )
        discard;
    fDistanceToExitPoint = min(fDistanceToExitPoint, fRayLength);

    float fCloudMass;
    float fIsecLenUSSpace = f2RayIsecs.y - f2RayIsecs.x;

    // Compute particle rendering attributes
    ComputeParticleRenderAttribs(   ParticleAttrs, 
                                    CellAttribs,
                                    ParticleLighting,
                                    f3CameraPos,
                                    f3ViewRay,
                                    f3EntryPointUSSpace, 
                                    f3ViewRayUSSpace,
                                    f3LightDirUSSpace,
                                    fDistanceToExitPoint,
                                    fDistanceToEntryPoint,
                                    f4Color
                                    );

    fTransparency = f4Color.w;
    f4Color.xyz  *= 1-fTransparency;
    fDistToCloud  = fTransparency < 0.99 ? fDistanceToEntryPoint : +FLT_MAX;
}




/*
 *  MAIN FUNC: 
 *  =========  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
void main() {    
	color = vec3(0.5, 0.0 ,0.0 );//rawcolor;//rawcolor;
}