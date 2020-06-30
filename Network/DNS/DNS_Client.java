//-----------------------------------------
// NAME: 1o
//-----------------------------------------
import java.net.*;
import java.util.*;
import java.io.*;

public class DNS_Client {
    private byte[] buffer = new byte[1024];

    private DatagramSocket ds = null;

//------------------------------------------------------
// DNS_Client
//
// PURPOSE: Constructor
//------------------------------------------------------
    public DNS_Client() throws Exception {
        ds = new DatagramSocket();
    }

//------------------------------------------------------
// send
//
// PURPOSE: send datagram packet
// INPUT PARAMETERS:
//  host: the server host
//  port: the port of server host
//  bytes: the message
//------------------------------------------------------
    public final DatagramPacket send(final String host, final int port, final byte[] bytes) throws IOException {
        DatagramPacket dp = new DatagramPacket(bytes, bytes.length, InetAddress.getByName(host), port);
        ds.send(dp);
        return dp;
    }
//------------------------------------------------------
// receive
//
// PURPOSE: receive message from server
// parameters.
//------------------------------------------------------
    public final byte[] receive(final String lhost, final int lport)
    throws Exception {
        DatagramPacket dp = new DatagramPacket(buffer, buffer.length);
        ds.receive(dp);
        byte[] otuput = new byte[dp.getLength()];
        System.arraycopy(dp.getData(), 0, otuput, 0, dp.getLength());
        return otuput;
    }

    public static void main(String[] args) throws Exception {
        Scanner input_string;

        boolean exit_loop;

        String dns_header;
        String dns_trailer;
        String name;
        String dns_socket;
        String prefix;
        String serverHost;

        int serverPort;
        int number_answer;

        serverPort = 53;
        serverHost = "8.8.8.8";
        prefix = "0x";
        dns_header = "0xC1 0x08 0x01 0x00 0x00 0x01 0x00 0x00 0x00 0x00 0x00 0x00";
        dns_trailer = "0x00 0x01 0x00 0x01";
        exit_loop = true;

        try {
            input_string = new Scanner(System.in);
            while (exit_loop) {
                int temp_index;

                temp_index = 0;

                System.out.println("Please type the domain name (q for exit programe):");

                // convert the input domain name to bytes
                name = input_string.nextLine();

                if (name == "q") {
                    exit_loop = false;
                    continue;
                }

                String[] temp = name.split("\\.");

                String[] resolve_name = new String[name.length() + 2];

                for (int index = 0; index < resolve_name.length && temp_index < temp.length; index++) {
                    if (temp[temp_index].length() < 16) {
                        resolve_name[index] = prefix + "0" + Integer.toHexString(temp[temp_index].length());
                    } else {
                        resolve_name[index] = prefix + Integer.toHexString(temp[temp_index].length());
                    }
                    index += temp[temp_index].length();
                    temp_index++;
                }

                resolve_name[resolve_name.length - 1] = prefix + "00";
                temp_index = 0;

                for (int index = 0; index < resolve_name.length;) {
                    if (index == 0) {
                        index++;
                        continue;
                    }

                    if (resolve_name[index] == null) {
                        for (int innerindex = 0; innerindex < temp[temp_index].length(); innerindex++) {
                            resolve_name[index] = prefix + Integer.toHexString(temp[temp_index].charAt(innerindex));
                            index++;
                        }
                    } else {
                        index++;
                        temp_index++;
                    }
                }

                // create dns socket in bytes
                dns_socket = dns_header;

                for (int index = 0; index < resolve_name.length; index++) {
                    dns_socket = dns_socket + " " + resolve_name[index];
                }

                dns_socket = dns_socket + " " + dns_trailer;

                String[] s = dns_socket.split(" ");

                byte[] b = new byte[s.length];

                for (int i = 0; i < s.length; i++) {
                    b[i] = (byte)Integer.parseInt(s[i].substring(2), 16);
                }

                // start to send and receive
                DNS_Client client = new DNS_Client();

                client.send(serverHost, serverPort, b);

                byte[] output = client.receive(serverHost, serverPort);

                int[] conevet_integer = new int[output.length];

                String[] response_code = new String[conevet_integer.length];

                // convert the response number
                for (int ii = 0; ii < conevet_integer.length; ii++) {
                    conevet_integer[ii] = output[ii] & 0xff;
                    if (conevet_integer[ii] < 16) {
                        response_code[ii] = prefix + "0" + Integer.toHexString(conevet_integer[ii]);
                    } else {
                        response_code[ii] = prefix + Integer.toHexString(conevet_integer[ii]);
                    }
                }

                // check the each field and output
                if ( (conevet_integer[3] % 8) == 0) {
                    System.out.println("\nThe length of the reply in bytes is " + output.length);
                    if ((conevet_integer[3] & 128) != 0) {
                        System.out.println("Recursion is available");
                    } else {
                        System.out.println("Recursion is not available");
                    }
                    System.out.println("Code of server response：" + Arrays.toString(response_code));
                    System.out.println("The number of answers in the answers field is " + conevet_integer[7]);

                    // convert the ip
                    int count = 0;
                    int count_ip = 0;
                    int offset = 0;
                    int index_count = 0;

                    offset = 12 + resolve_name.length + 2 + 2; // the first answer of the message

                    System.out.println("The list of IP addresses returned in the answers field are:");
                    while ( count < conevet_integer[7] && index_count < conevet_integer.length) {
                        if (conevet_integer[offset + 3] == 1) {
                            index_count = offset + 12;
                            System.out.println("    " + conevet_integer[index_count] + "." + conevet_integer[index_count + 1] + "." + conevet_integer[index_count + 2] + "." + conevet_integer[index_count + 3]);
                            count_ip++;
                        } else {
                            offset = offset + 11 + conevet_integer[offset + 11] + 1;
                        }
                        count ++;
                    }

                    System.out.println("The number of IP addresses found are " + count_ip);

                } else {
                    if ((conevet_integer[3] | 1) == conevet_integer[3]) {
                        System.out.println("Format error");
                    } else if ( (conevet_integer[3] | 2) == conevet_integer[3]) {
                        System.out.println("Server failure");
                    } else if ( (conevet_integer[3] | 3) == conevet_integer[3]) {
                        System.out.println("Name Error");
                    } else if ( (conevet_integer[3] | 4) == conevet_integer[3]) {
                        System.out.println("Not Implemented");
                    } else if ( (conevet_integer[3] | 5) == conevet_integer[3]) {
                        System.out.println("Refused");
                    }
                }
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
            e.printStackTrace();
        }
    }
}