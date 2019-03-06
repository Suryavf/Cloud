#version 330 core

// Interpolated values from the vertex shaders
in vec3 fragmentColor;
in vec2 fragmentCoordinate;

// Ouput data
out vec3 color;


/*
 *  GLOBAL VARIABLES: 
 *  ================  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#define OPTICAL_DEPTH_LUT_DIM vec4(64,32,64,32)

#define SAMPLE_4D_LUT(tex3DLUT, LUT_DIM, f4LUTCoords, fLOD, Result){ \
    vec3  f3UVW;                                                \
    f3UVW.xy = f4LUTCoords.xy;                                  \
    float fQSlice = f4LUTCoords.w * LUT_DIM.w - 0.5;            \
    float fQ0Slice = floor(fQSlice);                            \
    float fQWeight = fQSlice - fQ0Slice;                        \
                                                                \
    f3UVW.z = (fQ0Slice + f4LUTCoords.z) / LUT_DIM.w;           \
                                                                \
    Result = lerp(                                              \
        tex3DLUT.SampleLevel(samLinearWrap, f3UVW, fLOD),                                   \  // ---------------------------------------------------------- Tex
        /* frac() assures wraparound filtering of w coordinate*/                            \
        tex3DLUT.SampleLevel(samLinearWrap, frac(f3UVW + float3(0,0,1/LUT_DIM.w)), fLOD),   \  // ---------------------------------------------------------- Tex
        fQWeight);                                                                          \
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
/*
#define SAMPLE_4D_LUT(tex3D, LUT_DIM, f4LUTCoords, fLOD, Result){   \
    vec3  f3UVW;                                                    \
    f3UVW.xy = f4LUTCoords.xy;                                      \
    float fQSlice = f4LUTCoords.w * LUT_DIM.w - 0.5;                \
    float fQ0Slice = floor(fQSlice);                                \
    float fQWeight = fQSlice - fQ0Slice;                            \
                                                                    \
    f3UVW.z = (fQ0Slice + f4LUTCoords.z) / LUT_DIM.w;               \
                                                                    \
    Result = mix(  texture(tex3D,      f3UVW                            ).rgb,  \  // ---------------------------------------------------------- Tex
                   texture(tex3D, frac(f3UVW + float3(0,0,1/LUT_DIM.w)) ).rgb,  \  // ---------------------------------------------------------- Tex
                   fQWeight);                                                   \
}
*/
    SAMPLE_4D_LUT(g_tex3DParticleDensityLUT, OPTICAL_DEPTH_LUT_DIM, f4LUTCoords, 0, fNormalizedDensity); // ---------------------------------------------------------- Tex

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
    float fMultipleScattering = g_tex3DMultipleScatteringInParticleLUT.SampleLevel(samLinearWrap, f4MultipleScatteringLUTCoords.xyz, 0); // ---------------------------------------------------------- Tex
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