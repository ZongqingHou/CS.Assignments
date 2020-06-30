#include "fat_fs.h"
#include "fat.h"

struct FS_Instance_struct{
	int BytsPerSec;
	int SecPerClus; 
	int RsvdSecCnt;  
	int NumFATs;  
	int RootEntCnt; 
	int FATSz;
	int FAT_type;
	fatBS* fat_file;
	char* file_path;
};

FS_Instance *fs_create_instance(char *image_path){
	FS_Instance* result;
	fatBS* fat_bs;
	FILE* fat_file;
	int TotSec;
	int RootDirSectors;
	int DataSec;
	int CountofClusters;

	fat_bs = (fatBS*)malloc(sizeof(fatBS));
	result = (FS_Instance*)malloc(sizeof(FS_Instance));

	fat_file = fopen(image_path, "rb");

	fill_fatBS(fat_file, fat_bs);

	result->BytsPerSec = fat_bs->BPB_BytsPerSec;  
    result->SecPerClus = fat_bs->BPB_SecPerClus;  
    result->RsvdSecCnt = fat_bs->BPB_RsvdSecCnt;  
    result->NumFATs = fat_bs->BPB_NumFATs;  
    result->RootEntCnt = fat_bs->BPB_RootEntCnt;  
    result->FATSz = fat_bs->BPB_FATSz16;
    result->fat_file = fat_bs;
    result->file_path = image_path;

    RootDirSectors = (int)(((result->RootEntCnt*32)+(result->BytsPerSec-1))/result->BytsPerSec + 0.5);

	TotSec = fat_bs->BPB_TotSec16;

	DataSec = TotSec- (int)((result->RsvdSecCnt + result->NumFATs * result->FATSz + RootDirSectors)+0.5);
	CountofClusters = (int)(DataSec / result->SecPerClus +0.5);

	if(CountofClusters < 4085){
		result->FAT_type = 12;
	}else if(CountofClusters < 65525){
		result->FAT_type = 16;
	}else{
		result->FAT_type = 32;
	}

	fclose(fat_file);
    return result;
};

void fill_fatBS(FILE *fat_file, fatBS *fat_bs){
    int check;
    
    check = fseek(fat_file,0,SEEK_SET);
    if(check == -1)
        printf("fseed in fill_fatBS_struct failed!");
    
    check = fread(fat_bs,1,512,fat_file);
    if(check != 512)
        printf("fread in fill_fatBS_struct failed");
};

void print_info(FS_Instance *fat_fs){
	int total_byts;

	printf("Volume information for: %s\n", fat_fs->file_path);

	printf("Disk information:\n-----------------\n");
	printf("OEM Name: %s\n", fat_fs->fat_file->BS_OEMName);
	printf("Volume Label:\n");
	printf("File System Type (text): FAT%d\n", fat_fs->FAT_type);
  
  	printf("Media Type: %x (removable)\n", fat_fs->fat_file->BPB_Media);
  	total_byts = fat_fs->fat_file->BPB_TotSec16*fat_fs->BytsPerSec;
  	printf("Size: %d bytes (%2.2dMB)\n", total_byts, total_byts/(1024*1024));

  	printf("Disk geometry:\n-----------------\n");

  	printf("Bytes Per Sector: %d\n", fat_fs->BytsPerSec);
  	printf("Sectors Per Cluster: %d\n", fat_fs->SecPerClus);
  	printf("Total Sectors: %d\n", fat_fs->fat_file->BPB_TotSec16);
  	printf("Physical - Sectors per Track: %d\n", fat_fs->fat_file->BPB_SecPerTrk);
  	printf("Physical - Heads: %d\n", fat_fs->fat_file->BPB_NumHeads);

  	printf("File system info:\n-----------------\n");
  	printf("Volume ID: %d\n", fat_fs->fat_file->BPB_VolID);
  	printf("File System Type (calculated): FAT%d\n", fat_fs->FAT_type);
  	printf("FAT Size (sectors): %d\n", fat_fs->FATSz);
  	printf("Free space: %d bytes\n", 111);
};

FS_CurrentDir fs_get_root(FS_Instance *fat_fs){
	FS_CurrentDir result;

	result = (fat_fs->RsvdSecCnt + fat_fs->NumFATs * fat_fs->FATSz) * fat_fs->BytsPerSec;

	return result;
};

void print_dir(FS_Instance *fat_fs, FS_CurrentDir current_dir){
	int count_file;
	int count_dir;
	FILE* fat_file;

	fatEntry* fat_entry;
	fat_entry = (fatEntry*)malloc(sizeof(fat_entry));

	fat_file = fopen(fat_fs->file_path, "r");

	printf("%d", fat_fs->RootEntCnt);
	if(current_dir == fs_get_root(fat_fs)){
		int check;
		for(int i=0;i<fat_fs->RootEntCnt;i++){
			check = fseek(fat_file, current_dir, SEEK_SET);
			if(check == -1)
				printf("fseek failed!\n");

			check = fread(fat_entry,1,32,fat_file);
			if(check != 32)
				printf("fread failed!\n");

			if(fat_entry->DIR_Name[0] == 0x00){
				break;
			}

        	if((fat_entry->DIR_Attr&0x10) == 0 ) {
         	   	printf("%s        %d    A %s\n",fat_entry->DIR_Name,fat_entry->DIR_FileSize,fat_entry->DIR_WrtTime);
         	   	count_file++;  
     	  	} else {   
     	  		printf("%s        %d    A %s\n",fat_entry->DIR_Name,fat_entry->DIR_FileSize,fat_entry->DIR_WrtTime);
     	  		count_dir++;
			}
			current_dir += 32;
		}
		printf("       %d file(s)       %d dir(s)\n", count_file, count_dir);
	}else{
		int condition;
		int FAT_value;
		int data_area;

		data_area = fat_fs->BytsPerSec*(fat_fs->RsvdSecCnt + fat_fs->FATSz*fat_fs->NumFATs + (fat_fs->RootEntCnt*32 + fat_fs->BytsPerSec - 1)/fat_fs->BytsPerSec);

		if(fat_fs->FAT_type == 12){
			condition = 0x0FF8;
		}else if(fat_fs->FAT_type == 16){
			condition = 0x0FFF8;
		}else{
			condition = 0x0FFFFFF8;
		}

		while (FAT_value < condition) {
        	FAT_value = getFAT(fat_fs,current_dir);  
        	if((fat_fs->FAT_type == 12 && FAT_value == 0x0FF7)||(fat_fs->FAT_type == 16 && FAT_value == 0x0FFF7)||(fat_fs->FAT_type == 16 && FAT_value == 0x0FFFFFF7)){
				break;
			}
  			
  			int start_clus;
  			int count;

  			start_clus = data_area + (current_dir-2)*fat_fs->SecPerClus;

  			while(count<fat_fs->SecPerClus*fat_fs->BytsPerSec){
  				int check;
  				
        		check = fseek(fat_file,start_clus,SEEK_SET);  
        		if (check == -1){
            		printf("failed!");
            		continue;
            	} 
  
        		check = fread(fat_entry,1,32,fat_file);  
        		if (check != fat_fs->SecPerClus*fat_fs->BytsPerSec){  
            		printf("failed!");
            		continue;
            	}

            	if(fat_entry->DIR_Name[0] == 0x00){
					break;
				}
 
        		if((fat_entry->DIR_Attr&0x10) == 0 ) {
         	   		printf("%s        %d    A %d\n",fat_entry->DIR_Name,fat_entry->DIR_FileSize,fat_entry->DIR_WrtTime);
         	   		count_file++;  
     	  		} else {   
     	  			printf("%s        %d    A %d\n",fat_entry->DIR_Name,fat_entry->DIR_FileSize,fat_entry->DIR_WrtTime);
     	  			count_dir++;
				}
            	start_clus += 32;
            	count += 32;
  			}
  			current_dir = FAT_value;
  		}
  		printf("       %d file(s)       %d dir(s)\n", count_file, count_dir);
	}

	free(fat_entry);
	fclose(fat_file);
};

FS_CurrentDir change_dir(FS_Instance *fat_fs, FS_CurrentDir current_dir, char *path){
	int count_file;
	int count_dir;

	int check;
	char realName[12];
	fatEntry* fat_entry;
	FS_CurrentDir result;
	FILE* fat_file;

	fat_entry = (fatEntry*)malloc(32);
	fat_file = fopen(fat_fs->file_path, "rb");

	if(current_dir == fs_get_root(fat_fs)){

		for(int i=0;i<fat_fs->RootEntCnt;i++){	

			check = fseek(fat_file, current_dir, SEEK_SET);
			if(check == -1)
				printf("failed!");

			check = fread(fat_entry,1,32,fat_file);
			if(check != 32)
				printf("failed");

			if(fat_entry->DIR_Name[0] == 0x00){
				printf("1");
				break;
			}else{
				if(fat_entry->DIR_Name == path){
					printf("2");
					result = fat_entry->DIR_FstClusLO;
					break;
				}
			}
			current_dir += 32;
		}
	}else{
		int condition;
		int FAT_value;
		int data_area;

		data_area = fat_fs->BytsPerSec*(fat_fs->RsvdSecCnt + fat_fs->FATSz*fat_fs->NumFATs + (fat_fs->RootEntCnt*32 + fat_fs->BytsPerSec - 1)/fat_fs->BytsPerSec);

		if(fat_fs->FAT_type == 12){
			condition = 0x0FF8;
		}else if(fat_fs->FAT_type == 16){
			condition = 0x0FFF8;
		}else{
			condition = 0x0FFFFFF8;
		}

		while (FAT_value < condition) {
        	FAT_value = getFAT(fat_fs,current_dir);  
        	if((fat_fs->FAT_type == 12 && FAT_value == 0x0FF7)||(fat_fs->FAT_type == 16 && FAT_value == 0x0FFF7)||(fat_fs->FAT_type == 16 && FAT_value == 0x0FFFFFF7)){
				break;
			}
  			
  			int start_clus;
  			int count;
  			start_clus = data_area + (current_dir-2)*fat_fs->SecPerClus;

  			while(count<fat_fs->SecPerClus*fat_fs->BytsPerSec){
  				int check;
  				fatEntry* dir_entry;
        		check = fseek(fat_file,start_clus,SEEK_SET);  
        		if (check == -1){
            		printf("failed!");
            		continue;
            	} 
  
        		check = fread(dir_entry,1,32,fat_file);  
        		if (check != fat_fs->SecPerClus*fat_fs->BytsPerSec){  
            		printf("failed!");
            		continue;
            	}

            	if(dir_entry->DIR_Name[0] == 0x00){
					printf("no directory");
					break;
				}else{
					if(path == "."){
						return current_dir;
					}else if(path == ".."){
						result = dir_entry->DIR_FstClusHI;
						return result;
					}else if(path == dir_entry->DIR_Name){
						result = dir_entry->DIR_FstClusLO;
					}
				}
            	start_clus += 32;
            	count += 32;
  			}
  			current_dir = FAT_value;
  		}
	}
	return result;
}

void get_file(FS_Instance *fat_fs, FS_CurrentDir current_dir, char *path, char *local_path){
	FILE* fat_file;
	fatEntry* fat_entry;
	fat_file = fopen(fat_fs->file_path, "rb");

	if(current_dir == fs_get_root(fat_fs)){
		for(int i=0;i<fat_fs->RootEntCnt;i++){
			int check;

			check = fseek(fat_file, current_dir, SEEK_SET);
			if(check == -1)
				printf("failed!");

			check = fread(fat_entry,1,32,fat_file);
			if(check != 32)
				printf("failed");

			if(fat_entry->DIR_Name[0] == 0x00){
				printf("no directory");
				break;
			}else{
				if(fat_entry->DIR_Name == path){
					current_dir = fat_entry->DIR_FstClusLO;
					download(fat_fs, current_dir, path, local_path);
				}
			}
			current_dir += 32;
		}
	}else{
		int condition;
		int FAT_value;
		int data_area;

		data_area = fat_fs->BytsPerSec*(fat_fs->RsvdSecCnt + fat_fs->FATSz*fat_fs->NumFATs + (fat_fs->RootEntCnt*32 + fat_fs->BytsPerSec - 1)/fat_fs->BytsPerSec);

		if(fat_fs->FAT_type == 12){
			condition = 0x0FF8;
		}else if(fat_fs->FAT_type == 16){
			condition = 0x0FFF8;
		}else{
			condition = 0x0FFFFFF8;
		}

		while (FAT_value < condition) {
        	FAT_value = getFAT(fat_fs,current_dir);  
        	if((fat_fs->FAT_type == 12 && FAT_value == 0x0FF7)||(fat_fs->FAT_type == 16 && FAT_value == 0x0FFF7)||(fat_fs->FAT_type == 16 && FAT_value == 0x0FFFFFF7)){
				break;
			}
  			
  			int start_clus;
  			int count;
  			start_clus = data_area + (current_dir-2)*fat_fs->SecPerClus;

  			while(count<fat_fs->SecPerClus*fat_fs->BytsPerSec){
  				int check;
  				fatEntry* dir_entry;
        		check = fseek(fat_file,start_clus,SEEK_SET);  
        		if (check == -1){
            		printf("failed!");
            		continue;
            	} 
  
        		check = fread(dir_entry,1,32,fat_file);  
        		if (check != fat_fs->SecPerClus*fat_fs->BytsPerSec){  
            		printf("failed!");
            		continue;
            	}

            	if(dir_entry->DIR_Name[0] == 0x00){
					printf("no directory");
					break;
				}else{
					if(path == dir_entry->DIR_Name){
						download(fat_fs,current_dir,path,local_path);
					}
				}
            	start_clus += 32;
            	count += 32;
  			}
  			current_dir = FAT_value;
  		}
	}
	fclose(fat_file);
};

void download(FS_Instance *fat_fs, FS_CurrentDir current_dir, char *path, char *local_path){
	FILE* output_file;
	FILE* fat_file;
	fat_file = fopen(fat_fs->file_path, "rb");
	output_file = fopen(local_path, "r");

	int condition;
	int FAT_value;
	int data_area;

	data_area = fat_fs->BytsPerSec*(fat_fs->RsvdSecCnt + fat_fs->FATSz*fat_fs->NumFATs + (fat_fs->RootEntCnt*32 + fat_fs->BytsPerSec - 1)/fat_fs->BytsPerSec);

	if(fat_fs->FAT_type == 12){
		condition = 0x0FF8;
	}else if(fat_fs->FAT_type == 16){
		condition = 0x0FFF8;
	}else{
		condition = 0x0FFFFFF8;
	}

	while (FAT_value < condition) {
       	FAT_value = getFAT(fat_fs,current_dir);  
       	if((fat_fs->FAT_type == 12 && FAT_value == 0x0FF7)||(fat_fs->FAT_type == 16 && FAT_value == 0x0FFF7)||(fat_fs->FAT_type == 16 && FAT_value == 0x0FFFFFF7)){
			break;
		}
		

		fatEntry* dir_entry;
  			
  		int start_clus;
  		int count;
  		int check;
  		start_clus = data_area + (current_dir-2)*fat_fs->SecPerClus;

       	check = fseek(fat_file,start_clus,SEEK_SET);  
        if (check == -1){
           	printf("failed!");
           	continue;
        } 
  
        check = fread(dir_entry,1,512,fat_file);  
        if (check != fat_fs->SecPerClus*fat_fs->BytsPerSec){  
           	printf("failed!");
           	continue;
  		}
  		current_dir = FAT_value;
  		fputs(dir_entry, output_file);
	}
	fclose(fat_file);
	fclose(output_file);
};

void fs_cleanup(FS_Instance *fat_fs){
	free(fat_fs->fat_file);
	free(fat_fs);
};

uint16_t getFAT(FS_Instance* fat_fs, int Clus_num) {
	int fat_base;
	int fat_current_position;
	int check;
	int FATOffset;
	FILE* input;
	uint16_t FATClusEntry;
	uint16_t* FATClusEntry_ptr = &FATClusEntry; 

	FATClusEntry_ptr = (uint16_t*)malloc(sizeof(FATClusEntry_ptr));

	input = fopen(fat_fs->file_path,"rb");

	fat_base = fat_fs->RsvdSecCnt*fat_fs->BytsPerSec;

	if(fat_fs->FAT_type == 32){
		FATOffset = Clus_num * 4;
	}else if(fat_fs->FAT_type == 16){
		FATOffset = Clus_num *2;
	}else{
		FATOffset = Clus_num + (Clus_num/2);
	}

	fat_current_position = fat_base + FATOffset;
	
    check = fseek(input,fat_current_position,SEEK_SET);  
    if (check == -1){
        printf("failed!");
        exit;
    }
  
    check = fread(FATClusEntry_ptr,1,fat_fs->FAT_type,input);  
    if (check != fat_fs->FAT_type){
        printf("failed!");
        exit;
    }

	if(fat_fs->FAT_type == 12){
		if(Clus_num & 0x0001){
			FATClusEntry = FATClusEntry >> 4;
		}else{
			FATClusEntry = FATClusEntry << 4;
		}
	}else if(fat_fs->FAT_type == 16){
		return FATClusEntry;
	}else{
		FATClusEntry = FATClusEntry & 0x0FFFFFF;
	}

	fclose(input);

	return FATClusEntry;
};
