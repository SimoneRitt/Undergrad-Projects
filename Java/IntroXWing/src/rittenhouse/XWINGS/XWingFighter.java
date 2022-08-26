package rittenhouse.XWINGS;

import java.io.Serializable;
import java.util.Comparator;

import rittenhouse.ENUMS.*;
import rittenhouse.INTERFACES.Specifiable;

public abstract class XWingFighter implements Serializable, Specifiable {

	private static final long serialVersionUID = 1L;
	
	private String FighterID;
	private XModel FighterModel;
	private double FLength;
	private double FWidth;
	private double FHeight;
	private double FMass;
	private double FMaxSpeed;
	private XSensor FSensors;
	private XEngine FEngines;
	private XWeapon FPrimaryWpns;
	private XWeapon FSecondaryWpns;
	
	public XWingFighter(String fighterID, XModel fighterModel, 
			double fLength, double fWidth, double fHeight,
			double fMass, double fMaxSpeed) {
		this.FighterID = fighterID;
		this.FighterModel = fighterModel;
		this.FLength = fLength;
		this.FWidth = fWidth;
		this.FHeight = fHeight;
		this.FMass = fMass;
		this.FMaxSpeed = fMaxSpeed;
	}

	public String getFighterID() {
		return FighterID;
	}

	public void setFighterID(String fighterID) {
		FighterID = fighterID;
	}

	public XModel getFighterModel() {
		return FighterModel;
	}

	public void setFighterModel(XModel fighterModel) {
		FighterModel = fighterModel;
	}

	public double getFLength() {
		return FLength;
	}

	public void setFLength(double fLength) {
		FLength = fLength;
	}

	public double getFWidth() {
		return FWidth;
	}

	public void setFWidth(double fWidth) {
		FWidth = fWidth;
	}

	public double getFHeight() {
		return FHeight;
	}

	public void setFHeight(double fHeight) {
		FHeight = fHeight;
	}

	public double getFMass() {
		return FMass;
	}

	public void setFMass(double fMass) {
		FMass = fMass;
	}

	public double getFMaxSpeed() {
		return FMaxSpeed;
	}

	public void setFMaxSpeed(double fMaxSpeed) {
		FMaxSpeed = fMaxSpeed;
	}

	public XSensor getFSensors() {
		return FSensors;
	}

	public void setFSensors(XSensor fSensors) {
		FSensors = fSensors;
	}

	public XEngine getFEngines() {
		return FEngines;
	}

	public void setFEngines(XEngine fEngines) {
		FEngines = fEngines;
	}

	public XWeapon getFPrimaryWpns() {
		return FPrimaryWpns;
	}

	public void setFPrimaryWpns(XWeapon fPrimaryWpns) {
		FPrimaryWpns = fPrimaryWpns;
	}

	public XWeapon getFSecondaryWpns() {
		return FSecondaryWpns;
	}

	public void setFSecondaryWpns(XWeapon fSecondaryWpns) {
		FSecondaryWpns = fSecondaryWpns;
	}
	
	

}

class XCompare implements Comparator<XWingFighter> {

	@Override
	public int compare(XWingFighter F1, XWingFighter F2) { // compares XModel enumerations
		if(F1.getFighterModel().equals(F2.getFighterModel())) {
			return 1;
		}
		else {
			return 0;
		}
	}
	
}
