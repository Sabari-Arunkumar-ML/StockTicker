package main

import (
	"log"

	"github.com/emersion/go-imap"
	idle "github.com/emersion/go-imap-idle"
	"github.com/emersion/go-imap/client"
)

var (
	c                                    *client.Client
	err                                  error
	Last_Indexed_Mail_UID_Recorded_In_DB uint32 = 0
)

func getUnreadOrUnidexedEmails(lastIndexedUID uint32, mbox *imap.MailboxStatus) {

	criteria := imap.NewSearchCriteria()
	criteria.Uid = new(imap.SeqSet)
	criteria.Uid.AddRange(lastIndexedUID+1, mbox.Messages)
	uids, err := c.UidSearch(criteria)
	if err != nil {
		log.Fatal(err)
	}
	if len(uids) == 0 {
		log.Println("No new emails")
		return
	}
	seqset := new(imap.SeqSet)
	seqset.AddNum(uids...)

	for {
		messages := make(chan *imap.Message, 10)
		done := make(chan error, 1)
		go func() {
			done <- c.Fetch(seqset, []imap.FetchItem{imap.FetchEnvelope, imap.FetchUid}, messages)
		}()

		for msg := range messages {
			log.Println("Email Content", msg.Envelope.Subject)
			// record it after making the DB Transaction
			Last_Indexed_Mail_UID_Recorded_In_DB = msg.Uid
		}

		if err := <-done; err != nil {
			log.Fatal("Error fetching past messages", err)
		}
		// Strategies to be implemented to handle mutiple emails parallely
		break
	}
}

func readMailsInRealTime(mbox *imap.MailboxStatus) {
	idleClient := idle.NewClient(c)

	updates := make(chan client.Update)
	c.Updates = updates

	if ok, err := idleClient.SupportIdle(); err == nil && ok {
		stopped := false
		stop := make(chan struct{})
		done := make(chan error, 1)
		go func() {
			done <- idleClient.Idle(stop)
		}()

		for {
			select {
			case update := <-updates:
				log.Println("New update:", update)
				switch u := update.(type) {
				case *client.MailboxUpdate:
					log.Println("Mailbox updates found", u.Mailbox.UidNext, Last_Indexed_Mail_UID_Recorded_In_DB)
					if u.Mailbox.UidNext >= Last_Indexed_Mail_UID_Recorded_In_DB {
						// evaluate and find why the connection 'c' is struck, else we can pool 2 client connections to overcome this limitation
						getUnreadOrUnidexedEmails(Last_Indexed_Mail_UID_Recorded_In_DB, mbox)
					}
				}
				if !stopped {
					close(stop)
					stopped = true
				}
			case err := <-done:
				if err != nil {
					log.Fatal(err)
				}
				log.Println("Not idling anymore")
				return
			}
		}
	} else {
		// Fallback: call periodically c.Noop()
	}
}
func main() {
	c, err = client.DialTLS("imap.gmail.com:993", nil)
	if err != nil {
		log.Fatal(err)
	}
	log.Println("Connected to IMAP server")

	defer c.Logout()

	if err := c.Login("p.sabari1998@gmail.com", "xvfe bgda mdhn ophw"); err != nil {
		log.Fatal(err)
	}
	log.Println("Logged in")

	mbox, err := c.Select("INBOX", false)
	if err != nil {
		log.Fatal(err)
	}
	log.Println("Mailbox selected, number of messages:", mbox.Messages)

	if mbox.Messages == 0 {
		log.Println("No messages in mailbox")
		return
	}
	// specify last indexed message UUID and get any unreadMails
	getUnreadOrUnidexedEmails(33534, mbox)
	log.Println("Finished reading the mailbox; Proceeding to wait for active email consumption")

	readMailsInRealTime(mbox)

}
