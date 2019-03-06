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
#define OPTICAL_DEPTH_LUT_DIM vec4(64,32,64,32)
#define PARTICLE_SCALE_HEIGHT vec2(7994.f,1200.f)
#define EARTH_RADIUS          6360000.f

#define SAMPLE_4D_LUT(tex3D, LUT_DIM, f4LUTCoords, Result){   \
    vec3  f3UVW;                                                    \
    f3UVW.xy = f4LUTCoords.xy;                                      \
    float fQSlice = f4LUTCoords.w * LUT_DIM.w - 0.5;                \
    float fQ0Slice = floor(fQSlice);                                \
    float fQWeight = fQSlice - fQ0Slice;                            \
                                                                    \
    f3UVW.z = (fQ0Slice + f4LUTCoords.z) / LUT_DIM.w;               \
                                                                    \
    Result = mix(  texture3D(tex3D,      f3UVW                            ).r,  \ 
                   texture3D(tex3D, frac(f3UVW + float3(0,0,1/LUT_DIM.w)) ).r,  \ 
                   fQWeight);                                                   \
}

float _fCloudAltitude      = 3000.f;
float _fCloudThickness     =  700.f;
float _fAttenuationCoeff   =  0.07f; // Typical scattering coefficient lies in the range 0.01 - 0.1 m^-1
float _fParticleCutOffDist =  2e+5f;
// g_GlobalCloudAttribs.
float _fEarthRadius     = 6360000.f;


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

// GetRaySphereIntersection
// ........................
// http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
#define NO_INTERSECTIONS vec2(-1,-2)

void GetRaySphereIntersection(in vec3 f3RayOrigin,
                              in vec3 f3RayDirection,
                              in vec3 f3SphereCenter,
                              in float fSphereRadius,
                              out vec2 f2Intersections){
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
	vec4  f4MultipleScatteringLUTCoords = WorldParamsToParticleScatteringLUT(f3EntryPointUSSpace, f3ViewRayUSSpace, f3LightDirUSSpace, true);

    float fMultipleScattering = texture3D(g_tex3DMultipleScatteringInParticleLUT,  f4MultipleScatteringLUTCoords.xyz).r; 
	vec3  f3MultipleScattering = (1-fTransparency) * fMultipleScattering * f2SunLightAttenuation.y * ParticleLighting.f4SunLight.xyz;

	// Compute ambient light
	vec3 f3EarthCentre = vec3(0, -_fEarthRadius, 0);
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


/*
 *  MAIN FUNC: 
 *  =========  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
void main() {    
	color = vec3(0.5, 0.0 ,0.0 );//rawcolor;//rawcolor;
}