"""Skyfield lunar position calculator returning DataFrame."""
from datetime import datetime
import pandas as pd
from skyfield.api import load, Topos

def get_moon_phase_label(phase_deg):
    """Convert moon phase angle to descriptive label."""
    if 0 <= phase_deg < 22.5 or phase_deg >= 337.5:
        return "New Moon"
    elif 22.5 <= phase_deg < 67.5:
        return "Waxing Crescent"
    elif 67.5 <= phase_deg < 112.5:
        return "First Quarter"
    elif 112.5 <= phase_deg < 157.5:
        return "Waxing Gibbous"
    elif 157.5 <= phase_deg < 202.5:
        return "Full Moon"
    elif 202.5 <= phase_deg < 247.5:
        return "Waning Gibbous"
    elif 247.5 <= phase_deg < 292.5:
        return "Last Quarter"
    else:  # 292.5 <= phase_deg < 337.5
        return "Waning Crescent"


def get_tide_condition(moon_phase_deg, separation_deg):
    """Determine tide condition based on Moon phase and Sun-Moon separation."""
    # Spring tides: within 45° of alignment (0°) or opposition (180°)
    if (moon_phase_deg <= 45 or moon_phase_deg >= 315 or 
        135 <= moon_phase_deg <= 225):
        return "Spring Tide"
    # Neap tides: within 45° of 90° or 270°  
    elif (45 <= moon_phase_deg <= 135 or 
            225 <= moon_phase_deg <= 315):
        return "Neap Tide"
    else:
        return "Intermediate Tide"
    
    
def get_last_spring_neap_info(moon_phase_deg):
    """Calculate information about last spring/neap tide."""
    # Spring tides at 0° and 180°, Neap at 90° and 270°
    
    # Find the most recent spring and neap points in time
    # (not the nearest in phase space)
    
    if moon_phase_deg <= 180:
        # Last spring was at 0°, next spring at 180°
        last_spring_phase = 0
        days_since_spring = moon_phase_deg * 0.033
        last_neap_phase = 270 if moon_phase_deg > 270 else 90
        days_since_neap = (moon_phase_deg - last_neap_phase) % 360 * 0.033
    else:
        # Last spring was at 180°, next spring at 360° (0°)
        last_spring_phase = 180
        days_since_spring = (moon_phase_deg - 180) * 0.033
        last_neap_phase = 90 if moon_phase_deg > 90 else 270
        days_since_neap = (moon_phase_deg - last_neap_phase) % 360 * 0.033
    
    return {
        'days_since_last_spring': days_since_spring,
        'days_since_last_neap': days_since_neap,
        'last_spring_phase_deg': last_spring_phase,
        'last_neap_phase_deg': last_neap_phase
    }


def get_tidal_bulge_phase(moon_azimuth_deg, observer_longitude):
    """Determine if we're in the Moon-side or opposite-side tidal bulge."""
    # Moon-side bulge: Moon is near overhead (azimuth doesn't matter much, but generally)
    # Opposite-side bulge: Moon is ~180° away
    
    # Simplified: Use Moon's hour angle or just azimuth relative to location
    # For most purposes, we can use whether Moon is above/below horizon
    moon_above_horizon = moon_azimuth_deg > 0  # Actually need elevation, but azimuth gives rough idea
    
    if moon_above_horizon:
        return "moon_side_tide"  # Higher high water (usually)
    else:
        return "opposite_side_tide"  # Lower high water (usually)


def get_detailed_sun_phase(sun_altitude_deg):
    """More detailed Sun phase labels."""
    if sun_altitude_deg > 10:
        return "day_high"
    elif sun_altitude_deg > 0:
        return "day_low" 
    elif sun_altitude_deg > -1:  # Very narrow window for actual sunrise/set
        return "sunrise_sunset_transition"
    elif sun_altitude_deg > -6:
        return "civil_twilight"
    elif sun_altitude_deg > -12:
        return "nautical_twilight"
    elif sun_altitude_deg > -18:
        return "astronomical_twilight"
    else:
        return "night_true_dark"


def get_skyfield_positions(utc_time, latitude, longitude):
    """Get Moon and Sun positions using Skyfield and return as dictionary for DataFrame."""
    
    # Load ephemeris
    eph = load('de421.bsp')
    ts = load.timescale()
    
    # Convert to Skyfield time
    t = ts.utc(utc_time.year, utc_time.month, utc_time.day, 
            utc_time.hour, utc_time.minute, utc_time.second)
    
    # Set up Earth, Moon, Sun and observer
    earth = eph['earth']
    moon = eph['moon']
    sun = eph['sun']
    observer = earth + Topos(latitude_degrees=latitude, longitude_degrees=longitude)
    
    # Get apparent positions
    apparent_moon = observer.at(t).observe(moon).apparent()
    apparent_sun = observer.at(t).observe(sun).apparent()
    
    # Extract Moon data
    moon_ra, moon_dec, moon_distance = apparent_moon.radec()
    moon_alt, moon_az, _ = apparent_moon.altaz()
    
    # Extract Sun data
    sun_ra, sun_dec, sun_distance = apparent_sun.radec()
    sun_alt, sun_az, _ = apparent_sun.altaz()
    sun_elevation_deg = sun_alt.degrees
    sun_phase_label = get_detailed_sun_phase(sun_elevation_deg)
    
    # Get geocentric positions for comparison
    geocentric_moon = earth.at(t).observe(moon)
    moon_geo_ra, moon_geo_dec, moon_geo_distance = geocentric_moon.radec()
    
    geocentric_sun = earth.at(t).observe(sun)
    sun_geo_ra, sun_geo_dec, sun_geo_distance = geocentric_sun.radec()
    
    # Calculate Moon illumination and phase
    e = earth.at(t)
    s = e.observe(sun).apparent()
    m = e.observe(moon).apparent()
    
    # Get ecliptic longitudes for phase calculation
    from skyfield.framelib import ecliptic_frame
    _, slon, _ = s.frame_latlon(ecliptic_frame)
    _, mlon, _ = m.frame_latlon(ecliptic_frame)
    moon_phase = (mlon.degrees - slon.degrees) % 360.0
    
    # Add phase label
    moon_phase_label = get_moon_phase_label(moon_phase)
    
    # Get illumination percentage
    moon_illumination_percent = 100.0 * m.fraction_illuminated(sun)
    
    # Calculate angular separation between Sun and Moon as seen from observer
    sun_moon_separation = apparent_moon.separation_from(apparent_sun)
    
    # Calculate light times
    moon_light_time_minutes = moon_distance.km / 299792.458 / 60
    sun_light_time_minutes = sun_distance.km / 299792.458 / 60
    
    # Inside get_skyfield_positions function, add this before compiling results:
    tide_info = get_last_spring_neap_info(moon_phase)
    tide_condition = get_tide_condition(moon_phase, sun_moon_separation.degrees)
    tidal_bulge = get_tidal_bulge_phase(moon_az.degrees, longitude)
    
    # Compile results
    result = {
        # Moon Position data
        'moon_ra_icrf_deg': moon_ra.hours * 15,
        'moon_dec_icrf_deg': moon_dec.degrees,
        'moon_azimuth_deg': moon_az.degrees,
        'moon_elevation_deg': moon_alt.degrees,
        
        # Moon Distance data
        'moon_range_au': moon_distance.au,
        'moon_range_km': moon_distance.km,
        
        # Moon Geocentric data
        'moon_geocentric_ra_deg': moon_geo_ra.hours * 15,
        'moon_geocentric_dec_deg': moon_geo_dec.degrees,
        'moon_geocentric_distance_km': moon_geo_distance.km,
        
        # Sun Position data
        'sun_ra_icrf_deg': sun_ra.hours * 15,
        'sun_dec_icrf_deg': sun_dec.degrees,
        'sun_azimuth_deg': sun_az.degrees,
        'sun_elevation_deg': sun_alt.degrees,
        'sun_phase_label': sun_phase_label,
        
        # Sun Distance data
        'sun_range_au': sun_distance.au,
        'sun_range_km': sun_distance.km,
        
        # Sun Geocentric data
        'sun_geocentric_ra_deg': sun_geo_ra.hours * 15,
        'sun_geocentric_dec_deg': sun_geo_dec.degrees,
        'sun_geocentric_distance_km': sun_geo_distance.km,
        
        # Solar geometry - angular separation for tidal alignment
        'sun_moon_angular_separation_deg': sun_moon_separation.degrees,
        
        # Moon Illumination data
        'moon_phase_deg': moon_phase,
        'moon_phase_label': moon_phase_label,
        'moon_illumination_percent': moon_illumination_percent,
        
        # Tide information
        'expected_tide_height': "HHW" if tidal_bulge == "moon_side_tide" else "LHW",
        'tide_condition': tide_condition,
        'days_since_last_spring': tide_info['days_since_last_spring'],
        'days_since_last_neap': tide_info['days_since_last_neap'],
        'last_spring_phase_deg': tide_info['last_spring_phase_deg'],
        'last_neap_phase_deg': tide_info['last_neap_phase_deg'],
        
        # Time data
        'moon_light_time_minutes': moon_light_time_minutes,
        'sun_light_time_minutes': sun_light_time_minutes,
    }
    
    return result
