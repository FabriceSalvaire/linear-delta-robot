R = 400;
L = 500;
I = 1000;
difference() 
{
	difference()
	{
	translate([R,0,0])	sphere(r=L,$fn=50);
	{translate([-I,-I,-I+1]) cube ([2*I,2*I,I]);};
	}
	rotate(a=25)
	{
		translate([-10*L+100,-10*L,0])
		{
			cube([10*L,20*L,L]);
		}
	}
 }