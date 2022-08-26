package rittenhouse.TIESPT;

public class TiePilot {
	
	// variables
	
	private String IDNumber;
	private String Rank;
	private String PilotRating;
	
	// constructor
	
	public TiePilot(String IDNumber, String Rank, String PilotRating) {
		this.IDNumber = IDNumber;
		this.Rank = Rank;
		this.PilotRating = PilotRating;
		
	}
	
	// getters and setters
	
	public String getIDNumber() {
		return IDNumber;
	}

	public void setIDNumber(String iDNumber) {
		IDNumber = iDNumber;
	}

	public String getRank() {
		return Rank;
	}

	public void setRank(String rank) {
		Rank = rank;
	}

	public String getPilotRating() {
		return PilotRating;
	}

	public void setPilotRating(String pilotRating) {
		PilotRating = pilotRating;
	}
	
	// methods
	
	public void displaysInfo() {
		System.out.println("Imperial Pilot**********");
		System.out.println("ID: " + this.IDNumber);
		System.out.println("RANK: " + this.Rank);
		System.out.println("RATING: " + this.PilotRating);
	}

}
